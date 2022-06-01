import copy
import sys
import math

import numpy as np
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener, vec2)

from mod_gym import gym
from mod_gym.gym import spaces
from mod_gym.gym.utils import colorize, seeding, EzPickle

# This is simple 4-joints walker robot environment.
#
# There are two versions:
#
# - Normal, with slightly uneven terrain.
#
# - Hardcore with ladders, stumps, pitfalls.
#
# Reward is given for moving forward, total 300+ points up to the far end. If the robot falls,
# it gets -100. Applying motor torque costs a small amount of points, more optimal agent
# will get better score.
#
# Heuristic is provided for testing, it's also useful to get demonstrations to
# learn from. To run heuristic:
#
# python mod_gym/envs/box2d/bipedal_walker.py
#
# State consists of hull angle speed, angular velocity, horizontal speed, vertical speed,
# position of joints and joints angular speed, legs contact with ground, and 10 lidar
# rangefinder measurements to help to deal with the hardcore version. There's no coordinates
# in the state vector. Lidar is less useful in normal version, but it works.
#
# To solve the game you need to get 300 points in 1600 time steps.
#
# To solve hardcore version you need 300 points in 2000 time steps.
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

FPS    = 50
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

MOTORS_TORQUE = 80
SPEED_HIP     = 4
SPEED_KNEE    = 6
LIDAR_RANGE   = 160/SCALE

INITIAL_RANDOM = 5

HULL_POLY =[
    (-30,+9), (+6,+9), (+34,+1),
    (+34,-8), (-30,-8)
    ]
LEG_DOWN = -8/SCALE
LEG_W, LEG_H = 8/SCALE, 34/SCALE

VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN_STEP   = 14/SCALE
TERRAIN_LENGTH = 200     # in steps
TERRAIN_HEIGHT = VIEWPORT_H/SCALE/4
TERRAIN_GRASS    = 10    # low long are grass spots, in steps
TERRAIN_STARTPAD = 20    # in steps
FRICTION = 2.5

HULL_FD = fixtureDef(
                shape=polygonShape(vertices=[ (x/SCALE,y/SCALE) for x,y in HULL_POLY ]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0020,
                maskBits=0x001,  # collide only with ground
                restitution=0.0) # 0.99 bouncy

LEG_FD = fixtureDef(
                    shape=polygonShape(box=(LEG_W/2, LEG_H/2)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)

LOWER_FD = fixtureDef(
                    shape=polygonShape(box=(0.8*LEG_W/2, LEG_H/2)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)

class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        if self.env.hull==contact.fixtureA.body or self.env.hull==contact.fixtureB.body:
            self.env.game_over = True
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = True
    def EndContact(self, contact):
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = False

class BipedalWalker(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    hardcore = False

    def __init__(self):
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None

        self.world = Box2D.b2World()
        self.terrain = None
        self.hull = None

        self.prev_shaping = None

        self.fd_polygon = fixtureDef(
                        shape = polygonShape(vertices=
                        [(0, 0),
                         (1, 0),
                         (1, -1),
                         (0, -1)]),
                        friction = FRICTION)

        self.fd_edge = fixtureDef(
                    shape = edgeShape(vertices=
                    [(0, 0),
                     (1, 1)]),
                    friction = FRICTION,
                    categoryBits=0x0001,
                )

        self.reset()

        high = np.array([np.inf] * 24)
        self.action_space = spaces.Box(np.array([-1, -1, -1, -1]), np.array([1, 1, 1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.terrain: return
        self.world.contactListener = None
        for t in self.terrain:
            self.world.DestroyBody(t)
        self.terrain = []
        self.world.DestroyBody(self.hull)
        self.hull = None
        for leg in self.legs:
            self.world.DestroyBody(leg)
        self.legs = []
        self.joints = []

    def _generate_terrain(self, hardcore, terrain_x=None, terrain_y=None, terrain_type_poly=None):
        GRASS, STUMP, STAIRS, PIT, _STATES_ = range(5)
        state    = GRASS
        velocity = 0.0
        y        = TERRAIN_HEIGHT
        counter  = TERRAIN_STARTPAD
        oneshot  = False
        self.terrain   = []
        self.terrain_x = []
        self.terrain_y = []
        # self.fd_polygon_poly_list = []
        self.terrain_type_poly = []
        for i in range(TERRAIN_LENGTH):
            x = i*TERRAIN_STEP
            self.terrain_x.append(x)

            if terrain_type_poly is not None: state = terrain_type_poly[i][0]

            if state==GRASS and not oneshot:
                velocity = 0.8*velocity + 0.01*np.sign(TERRAIN_HEIGHT - y)
                if i > TERRAIN_STARTPAD: velocity += self.np_random.uniform(-1, 1)/SCALE   #1
                y += velocity
                self.terrain_type_poly.append((GRASS, None, x, y))
            elif state==PIT and oneshot:
                counter = self.np_random.randint(3, 5)
                if terrain_type_poly is not None:
                    poly = terrain_type_poly[i][1]  # next(poly_iterator)
                else:
                    poly = [
                        (x,              y),
                        (x+TERRAIN_STEP, y),
                        (x+TERRAIN_STEP, y-4*TERRAIN_STEP),
                        (x,              y-4*TERRAIN_STEP),
                        ]
                # self.fd_polygon_poly_list.append(poly)
                self.terrain_type_poly.append((PIT, poly, x, y))
                self.fd_polygon.shape.vertices=poly
                t = self.world.CreateStaticBody(
                    fixtures = self.fd_polygon)
                t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                self.terrain.append(t)

                if terrain_type_poly is not None:
                    pit_poly = terrain_type_poly[i][1]  # next(poly_iterator)
                else:
                    pit_poly = [(p[0]+TERRAIN_STEP*counter,p[1]) for p in poly]

                # self.fd_polygon_poly_list.append(pit_poly)
                self.terrain_type_poly.append((PIT, pit_poly, x, y))
                self.fd_polygon.shape.vertices = pit_poly

                t = self.world.CreateStaticBody(
                    fixtures = self.fd_polygon)
                t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                self.terrain.append(t)
                counter += 2
                original_y = y

            elif state==PIT and not oneshot:
                y = original_y
                if counter > 1:
                    y -= 4*TERRAIN_STEP
                self.terrain_type_poly.append((PIT, None, x, y))

            elif state==STUMP and oneshot:
                counter = self.np_random.randint(1, 3)
                if terrain_type_poly is not None:
                    poly = terrain_type_poly[i][1]  # next(poly_iterator)
                else:
                    poly = [
                        (x,                      y),
                        (x+counter*TERRAIN_STEP, y),
                        (x+counter*TERRAIN_STEP, y+counter*TERRAIN_STEP),
                        (x,                      y+counter*TERRAIN_STEP),
                        ]
                # self.fd_polygon_poly_list.append(poly)
                self.terrain_type_poly.append((STUMP, poly, x, y))
                self.fd_polygon.shape.vertices=poly
                t = self.world.CreateStaticBody(
                    fixtures = self.fd_polygon)
                t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                self.terrain.append(t)

            elif state==STAIRS and oneshot:
                stair_height = +1 if self.np_random.rand() > 0.5 else -1
                stair_width = self.np_random.randint(4, 5)
                stair_steps = self.np_random.randint(3, 5)
                original_y = y
                poly_list = []
                # STAIR_STEPS OVERRIDEN HERE
                if terrain_type_poly is not None:
                    stair_steps = len(terrain_type_poly[i][1])
                for s in range(stair_steps):
                    # STAIR_HEIGHT AND STAIR_WIDTH OVERRIDEN HERE
                    if terrain_type_poly is not None:
                        poly = terrain_type_poly[i][1][s]
                    else:
                        poly = [
                            (x+(    s*stair_width)*TERRAIN_STEP, y+(   s*stair_height)*TERRAIN_STEP),
                            (x+((1+s)*stair_width)*TERRAIN_STEP, y+(   s*stair_height)*TERRAIN_STEP),
                            (x+((1+s)*stair_width)*TERRAIN_STEP, y+(-1+s*stair_height)*TERRAIN_STEP),
                            (x+(    s*stair_width)*TERRAIN_STEP, y+(-1+s*stair_height)*TERRAIN_STEP),
                            ]
                    poly_list.append(poly)
                    # self.fd_polygon_poly_list.append(poly)
                    self.fd_polygon.shape.vertices=poly
                    t = self.world.CreateStaticBody(
                        fixtures = self.fd_polygon)
                    t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                    self.terrain.append(t)

                self.terrain_type_poly.append((STAIRS, poly_list, x, y))
                counter = stair_steps*stair_width

            elif state==STAIRS and not oneshot:
                s = stair_steps*stair_width - counter - stair_height
                n = s/stair_width
                y = original_y + (n*stair_height)*TERRAIN_STEP
                self.terrain_type_poly.append((STAIRS, None, x, y))
            else:
                self.terrain_type_poly.append((None, None, None, None))

            oneshot = False
            self.terrain_y.append(y)
            counter -= 1
            if counter==0:
                counter = self.np_random.randint(TERRAIN_GRASS/2, TERRAIN_GRASS)
                if state==GRASS and hardcore:
                    state = self.np_random.randint(1, _STATES_)
                    oneshot = True
                else:
                    state = GRASS
                    oneshot = True

        ## OVERRIDE TERRAIN_X and TERRAIN_Y HERE
        if terrain_x is not None:
            self.terrain_x = terrain_x
        if terrain_y is not None:
            self.terrain_y = terrain_y

        self.terrain_poly = []
        for i in range(TERRAIN_LENGTH-1):
            poly = [
                (self.terrain_x[i],   self.terrain_y[i]),
                (self.terrain_x[i+1], self.terrain_y[i+1])
                ]
            self.fd_edge.shape.vertices=poly
            t = self.world.CreateStaticBody(
                fixtures = self.fd_edge)
            color = (0.3, 1.0 if i%2==0 else 0.8, 0.3)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (0.4, 0.6, 0.3)
            poly += [ (poly[1][0], 0), (poly[0][0], 0) ]
            self.terrain_poly.append( (poly, color) )
        self.terrain.reverse()

    def _generate_clouds(self):
        # Sorry for the clouds, couldn't resist
        self.cloud_poly   = []
        for i in range(TERRAIN_LENGTH//20):
            x = self.np_random.uniform(0, TERRAIN_LENGTH)*TERRAIN_STEP
            y = VIEWPORT_H/SCALE*3/4
            poly = [
                (x+15*TERRAIN_STEP*math.sin(3.14*2*a/5)+self.np_random.uniform(0,5*TERRAIN_STEP),
                 y+ 5*TERRAIN_STEP*math.cos(3.14*2*a/5)+self.np_random.uniform(0,5*TERRAIN_STEP) )
                for a in range(5) ]
            x1 = min( [p[0] for p in poly] )
            x2 = max( [p[0] for p in poly] )
            self.cloud_poly.append( (poly,x1,x2) )

    def create_hull(self, pos, angle=None, linVel=None, angVel=None):
        self.hull = self.world.CreateDynamicBody(
            position=pos,
            fixtures=HULL_FD
        )

        if angle is not None:
            self.hull.angle = angle
        if linVel is not None:
            self.hull.linearVelocity = linVel
        if angVel is not None:
            self.hull.angularVelocity = angVel

        self.hull.color1 = (0.5, 0.4, 0.9)
        self.hull.color2 = (0.3, 0.3, 0.5)

    def create_legs(self, i, leg_positions, leg_vels, leg_angles, leg_ang_vels, leg_contacts, joint_motor_speeds, joint_max_motor_torques):
        pos = leg_positions[i+1]
        pos_x, pos_y = pos
        leg = self.world.CreateDynamicBody(
            position=(pos_x, pos_y),  # - LEG_H/2 - LEG_DOWN),
            angle=(leg_angles[i+1]),  # (i*0.05 + self.hull.angle),
            fixtures=LEG_FD
        )
        if leg_vels is not None: leg.linearVelocity = leg_vels[i+1]
        if leg_ang_vels is not None: leg.angularVelocity = leg_ang_vels[i+1]
        leg.color1 = (0.6-i/10., 0.3-i/10., 0.5-i/10.)
        leg.color2 = (0.4-i/10., 0.2-i/10., 0.3-i/10.)
        m_speed = joint_motor_speeds[i+1]
        mm_torque = joint_max_motor_torques[i+1]
        rjd = revoluteJointDef(
            bodyA=self.hull,
            bodyB=leg,
            localAnchorA=(0, LEG_DOWN),
            localAnchorB=(0, LEG_H/2),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=mm_torque,
            motorSpeed = m_speed,
            lowerAngle = -0.8,
            upperAngle = 1.1,
            referenceAngle=i*0.05,
        )
        leg.ground_contact = leg_contacts[i+1]
        self.legs.append(leg)
        self.joints.append(self.world.CreateJoint(rjd))
        pos = leg_positions[i+2]
        pos_x, pos_y = pos
        lower = self.world.CreateDynamicBody(
            position=(pos_x, pos_y),  # - LEG_H*3/2 - LEG_DOWN),
            angle=(leg_angles[i+2]),  # (i*0.05 + leg.angle),
            fixtures=LOWER_FD
        )
        if leg_ang_vels is not None: lower.angularVelocity = leg_ang_vels[i+2]
        if leg_vels is not None: lower.linearVelocity = leg_vels[i+2]
        lower.color1 = (0.6-i/10., 0.3-i/10., 0.5-i/10.)
        lower.color2 = (0.4-i/10., 0.2-i/10., 0.3-i/10.)
        m_speed = joint_motor_speeds[i+2]
        mm_torque = joint_max_motor_torques[i+2]
        rjd = revoluteJointDef(
            bodyA=leg,
            bodyB=lower,
            localAnchorA=(0, -LEG_H/2),
            localAnchorB=(0, LEG_H/2),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=mm_torque,
            motorSpeed = m_speed,
            lowerAngle = -1.6,
            upperAngle = -0.1,
            referenceAngle=i*0.05,
        )
        lower.ground_contact = leg_contacts[i+2]
        self.legs.append(lower)
        self.joints.append(self.world.CreateJoint(rjd))


    def reset(self, hi_lvl_state=None, rand_state=None):
        self._destroy()
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.game_over = False
        self.prev_shaping = None
        self.scroll = 0.0
        self.lidar_render = 0

        W = VIEWPORT_W/SCALE
        H = VIEWPORT_H/SCALE

        self._generate_clouds()

        if hi_lvl_state is not None:
            if len(hi_lvl_state) == 17: hi_lvl_state = hi_lvl_state[:-3]  # this line is required as bipedal seeds include useless lidar info
            hull_pos, hull_vel, hull_ang, hull_angVel, \
                leg_positions, leg_angles, leg_contacts, \
                joint_motor_speeds, joint_max_motor_torques, \
                leg_ang_vels, leg_vels, \
                terrain_x, terrain_y, terrain_type_poly = hi_lvl_state
                # lidar_p1s, lidar_p2s, lidar_frctn = hi_lvl_state
            self.scroll = hull_pos[0] - VIEWPORT_W/SCALE/5
            if rand_state is not None: self.np_random.set_state(rand_state)  # for Bug oracles we dont want to restore the state on the particular random state but in SeedBug Oracle we want that.
        else:
            init_x = TERRAIN_STEP*TERRAIN_STARTPAD/2
            init_y = TERRAIN_HEIGHT+2*LEG_H
            joint_motor_speeds = [-1, 1, 1, 1]
            joint_max_motor_torques = [MOTORS_TORQUE] * 4
            leg_contacts = [False] * 4
            leg_positions = [(init_x, init_y  - LEG_H/2 - LEG_DOWN), (init_x, init_y - LEG_H*3/2 - LEG_DOWN), (init_x, init_y  - LEG_H/2 - LEG_DOWN), (init_x, init_y - LEG_H*3/2 - LEG_DOWN)]
            hull_pos = (init_x, init_y)
            hull_ang, hull_vel, hull_angVel = None, None, None
            terrain_x, terrain_y, terrain_type_poly = None, None, None
            leg_angles = [-0.05, -0.05, 0.05, 0.05]
            leg_ang_vels, leg_vels = None, None

        lidar_p1s, lidar_p2s, lidar_frctn = None, None, None
        self._generate_terrain(self.hardcore, terrain_x, terrain_y, terrain_type_poly)
        self.create_hull(hull_pos, angle=hull_ang, linVel=hull_vel, angVel=hull_angVel)
        if hi_lvl_state is None:
            self.hull.ApplyForceToCenter((self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), 0), True)

        self.legs = []
        self.joints = []
        self.create_legs(-1, leg_positions, leg_vels, leg_angles, leg_ang_vels, leg_contacts, joint_motor_speeds, joint_max_motor_torques)
        self.create_legs(+1, leg_positions, leg_vels, leg_angles, leg_ang_vels, leg_contacts, joint_motor_speeds, joint_max_motor_torques)

        self.drawlist = self.terrain + self.legs + [self.hull]

        class LidarCallback(Box2D.b2.rayCastCallback):
            def ReportFixture(self, fixture, point, normal, fraction):
                if (fixture.filterData.categoryBits & 1) == 0:
                    return -1
                self.p2 = point
                self.fraction = fraction
                return fraction
        self.lidar = [LidarCallback() for _ in range(10)]

        if lidar_p1s is not None:
            for i in range(10):
                self.lidar[i].fraction = lidar_frctn[i]
                self.lidar[i].p1 = vec2(lidar_p1s[i])
                self.lidar[i].p2 = lidar_p2s[i]
        else:
            for i in range(10):
                self.lidar[i].fraction = 1.0
                self.lidar[i].p1 = self.hull.position
                self.lidar[i].p2 = (
                    self.hull.position[0] + math.sin(1.5*i/10.0)*LIDAR_RANGE,
                    self.hull.position[1] - math.cos(1.5*i/10.0)*LIDAR_RANGE)
                self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

        if hi_lvl_state is None:
            return self.step(np.array([0,0,0,0]))[0]  # self.step(np.array([0, 0]) if self.continuous else 0)[0]
        else:
            observation, _, _ = self.get_state()
            return observation
        # return self.step(np.array([0,0,0,0]))[0]


    def get_state(self):
        hull_pos = self.hull.position
        hull_vel = self.hull.linearVelocity
        nn_state = [
            self.hull.angle,        # Normal angles up to 0.5 here, but sure more is possible.
            2.0*self.hull.angularVelocity/FPS,
            0.3*hull_vel.x*(VIEWPORT_W/SCALE)/FPS,  # Normalized to get -1..1 range
            0.3*hull_vel.y*(VIEWPORT_H/SCALE)/FPS,
            self.joints[0].angle,   # This will give 1.1 on high up, but it's still OK (and there should be spikes on hiting the ground, that's normal too)
            self.joints[0].speed / SPEED_HIP,
            self.joints[1].angle + 1.0,
            self.joints[1].speed / SPEED_KNEE,
            1.0 if self.legs[1].ground_contact else 0.0,
            self.joints[2].angle,
            self.joints[2].speed / SPEED_HIP,
            self.joints[3].angle + 1.0,
            self.joints[3].speed / SPEED_KNEE,
            1.0 if self.legs[3].ground_contact else 0.0
        ]
        nn_state += [l.fraction for l in self.lidar]
        assert len(nn_state)==24

        hull_angle = self.hull.angle
        hull_ang_vel = self.hull.angularVelocity
        leg_positions = [(leg.position.x, leg.position.y) for leg in self.legs]
        leg_contacts = [leg.ground_contact for leg in self.legs]
        joint_motor_speeds = [jnt.motorSpeed for jnt in self.joints]
        joint_max_motor_torques = [jnt.GetMaxMotorTorque() for jnt in self.joints]
        leg_angles = [lg.angle for lg in self.legs]
        leg_vels = [(leg.linearVelocity.x, leg.linearVelocity.y) for leg in self.legs]
        leg_ang_vels = [leg.angularVelocity for leg in self.legs]

        terrain_x = self.terrain_x
        terrain_y = self.terrain_y
        terrain_type_poly = self.terrain_type_poly
        # lidar_p1s, lidar_p2s, lidar_frctn = [], [], []
        # for ldr in self.lidar:
        #     lidar_p1s.append((ldr.p1.x, ldr.p1.y))
        #     lidar_p2s.append(ldr.p2)
        #     lidar_frctn.append(ldr.fraction)

        hi_lvl_state = [(hull_pos.x, hull_pos.y), (hull_vel.x, hull_vel.y), hull_angle, hull_ang_vel,
                        leg_positions, leg_angles, leg_contacts, joint_motor_speeds, joint_max_motor_torques,
                        leg_ang_vels, leg_vels, terrain_x, terrain_y, terrain_type_poly]  # , lidar_p1s, lidar_p2s, lidar_frctn]

        return nn_state, hi_lvl_state, self.np_random.get_state()  # current random state

    # @profile
    def step(self, action):
        #self.hull.ApplyForceToCenter((0, 20), True) -- Uncomment this to receive a bit of stability help
        control_speed = False  # Should be easier as well
        if control_speed:
            self.joints[0].motorSpeed = float(SPEED_HIP  * np.clip(action[0], -1, 1))
            self.joints[1].motorSpeed = float(SPEED_KNEE * np.clip(action[1], -1, 1))
            self.joints[2].motorSpeed = float(SPEED_HIP  * np.clip(action[2], -1, 1))
            self.joints[3].motorSpeed = float(SPEED_KNEE * np.clip(action[3], -1, 1))
        else:
            self.joints[0].motorSpeed     = float(SPEED_HIP     * np.sign(action[0]))
            self.joints[0].maxMotorTorque = float(MOTORS_TORQUE * np.core.umath.minimum(np.abs(action[0]), 1))  # np.clip(, 0, 1))
            self.joints[1].motorSpeed     = float(SPEED_KNEE    * np.sign(action[1]))
            self.joints[1].maxMotorTorque = float(MOTORS_TORQUE * np.core.umath.minimum(np.abs(action[1]), 1))  # np.clip(np.abs(action[1]), 0, 1))
            self.joints[2].motorSpeed     = float(SPEED_HIP     * np.sign(action[2]))
            self.joints[2].maxMotorTorque = float(MOTORS_TORQUE * np.core.umath.minimum(np.abs(action[2]), 1))  # np.clip(np.abs(action[2]), 0, 1))
            self.joints[3].motorSpeed     = float(SPEED_KNEE    * np.sign(action[3]))
            self.joints[3].maxMotorTorque = float(MOTORS_TORQUE * np.core.umath.minimum(np.abs(action[3]), 1))  # np.clip(np.abs(action[3]), 0, 1))

        self.world.Step(1.0/FPS, 6*30, 2*30)

        pos = self.hull.position
        vel = self.hull.linearVelocity

        for i in range(10):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin(1.5*i/10.0)*LIDAR_RANGE,
                pos[1] - math.cos(1.5*i/10.0)*LIDAR_RANGE)
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

        state = [
            self.hull.angle,        # Normal angles up to 0.5 here, but sure more is possible.
            2.0*self.hull.angularVelocity/FPS,
            0.3*vel.x*(VIEWPORT_W/SCALE)/FPS,  # Normalized to get -1..1 range
            0.3*vel.y*(VIEWPORT_H/SCALE)/FPS,
            self.joints[0].angle,   # This will give 1.1 on high up, but it's still OK (and there should be spikes on hiting the ground, that's normal too)
            self.joints[0].speed / SPEED_HIP,
            self.joints[1].angle + 1.0,
            self.joints[1].speed / SPEED_KNEE,
            1.0 if self.legs[1].ground_contact else 0.0,
            self.joints[2].angle,
            self.joints[2].speed / SPEED_HIP,
            self.joints[3].angle + 1.0,
            self.joints[3].speed / SPEED_KNEE,
            1.0 if self.legs[3].ground_contact else 0.0
            ]
        state += [l.fraction for l in self.lidar]
        assert len(state)==24

        self.scroll = pos.x - VIEWPORT_W/SCALE/5

        shaping  = 130*pos[0]/SCALE   # moving forward is a way to receive reward (normalized to get 300 on completion)
        shaping -= 5.0*abs(state[0])  # keep head straight, other than that and falling, any behavior is unpunished

        reward = 0
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        for a in action:
            reward -= 0.00035 * MOTORS_TORQUE * np.core.umath.minimum(np.abs(a), 1)  # np.core.umath.minimum(np.abs(a), 1), 0)  # np.clip(np.abs(a), 0, 1)
            # normalized to about -50.0 using heuristic, more optimal agent should spend less

        done = False
        if self.game_over or pos[0] < 0:
            reward = -100
            done   = True
        if pos[0] > (TERRAIN_LENGTH-TERRAIN_GRASS)*TERRAIN_STEP:
            done   = True

        return np.array(state), reward, done, {}

    def my_clip(self, val):
        return 1 if val > 1 else val

    def get_sign(self, val):
        if val < 0:
            return -1
        elif val > 0:
            return 1
        else:
            return 0

    def render(self, mode='human'):
        from mod_gym.gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
        self.viewer.set_bounds(self.scroll, VIEWPORT_W/SCALE + self.scroll, 0, VIEWPORT_H/SCALE)

        self.viewer.draw_polygon( [
            (self.scroll,                  0),
            (self.scroll+VIEWPORT_W/SCALE, 0),
            (self.scroll+VIEWPORT_W/SCALE, VIEWPORT_H/SCALE),
            (self.scroll,                  VIEWPORT_H/SCALE),
            ], color=(0.9, 0.9, 1.0) )
        for poly,x1,x2 in self.cloud_poly:
            if x2 < self.scroll/2: continue
            if x1 > self.scroll/2 + VIEWPORT_W/SCALE: continue
            self.viewer.draw_polygon( [(p[0]+self.scroll/2, p[1]) for p in poly], color=(1,1,1))
        for poly, color in self.terrain_poly:
            if poly[1][0] < self.scroll: continue
            if poly[0][0] > self.scroll + VIEWPORT_W/SCALE: continue
            self.viewer.draw_polygon(poly, color=color)

        self.lidar_render = (self.lidar_render+1) % 100
        i = self.lidar_render
        # if i < 2*len(self.lidar):
        #     l = self.lidar[i] if i < len(self.lidar) else self.lidar[len(self.lidar)-i-1]
        #     self.viewer.draw_polyline( [l.p1, l.p2], color=(1,0,0), linewidth=1 )

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        flagy1 = TERRAIN_HEIGHT
        flagy2 = flagy1 + 50/SCALE
        x = TERRAIN_STEP*3
        self.viewer.draw_polyline( [(x, flagy1), (x, flagy2)], color=(0,0,0), linewidth=2 )
        f = [(x, flagy2), (x, flagy2-10/SCALE), (x+25/SCALE, flagy2-5/SCALE)]
        self.viewer.draw_polygon(f, color=(0.9,0.2,0) )
        self.viewer.draw_polyline(f + [f[0]], color=(0,0,0), linewidth=2 )

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

class BipedalWalkerHardcore(BipedalWalker):
    hardcore = True

if __name__=="__main__":
    # Heurisic: suboptimal, have no notion of balance.
    env = BipedalWalkerHardcore()  # BipedalWalker()
    env.reset()
    steps = 0
    total_reward = 0
    a = np.array([0.0, 0.0, 0.0, 0.0])
    STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1,2,3
    SPEED = 0.29  # Will fall forward on higher speed
    state = STAY_ON_ONE_LEG
    moving_leg = 0
    supporting_leg = 1 - moving_leg
    SUPPORT_KNEE_ANGLE = +0.1
    supporting_knee_angle = SUPPORT_KNEE_ANGLE
    while True:
        s, r, done, info = env.step(a)
        total_reward += r
        if steps % 20 == 0 or done:
            print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            print("hull " + str(["{:+0.2f}".format(x) for x in s[0:4] ]))
            print("leg0 " + str(["{:+0.2f}".format(x) for x in s[4:9] ]))
            print("leg1 " + str(["{:+0.2f}".format(x) for x in s[9:14]]))
        steps += 1

        contact0 = s[8]
        contact1 = s[13]
        moving_s_base = 4 + 5*moving_leg
        supporting_s_base = 4 + 5*supporting_leg

        hip_targ  = [None,None]   # -0.8 .. +1.1
        knee_targ = [None,None]   # -0.6 .. +0.9
        hip_todo  = [0.0, 0.0]
        knee_todo = [0.0, 0.0]

        if state==STAY_ON_ONE_LEG:
            hip_targ[moving_leg]  = 1.1
            knee_targ[moving_leg] = -0.6
            supporting_knee_angle += 0.03
            if s[2] > SPEED: supporting_knee_angle += 0.03
            supporting_knee_angle = min( supporting_knee_angle, SUPPORT_KNEE_ANGLE )
            knee_targ[supporting_leg] = supporting_knee_angle
            if s[supporting_s_base+0] < 0.10: # supporting leg is behind
                state = PUT_OTHER_DOWN
        if state==PUT_OTHER_DOWN:
            hip_targ[moving_leg]  = +0.1
            knee_targ[moving_leg] = SUPPORT_KNEE_ANGLE
            knee_targ[supporting_leg] = supporting_knee_angle
            if s[moving_s_base+4]:
                state = PUSH_OFF
                supporting_knee_angle = min( s[moving_s_base+2], SUPPORT_KNEE_ANGLE )
        if state==PUSH_OFF:
            knee_targ[moving_leg] = supporting_knee_angle
            knee_targ[supporting_leg] = +1.0
            if s[supporting_s_base+2] > 0.88 or s[2] > 1.2*SPEED:
                state = STAY_ON_ONE_LEG
                moving_leg = 1 - moving_leg
                supporting_leg = 1 - moving_leg

        if hip_targ[0]: hip_todo[0] = 0.9*(hip_targ[0] - s[4]) - 0.25*s[5]
        if hip_targ[1]: hip_todo[1] = 0.9*(hip_targ[1] - s[9]) - 0.25*s[10]
        if knee_targ[0]: knee_todo[0] = 4.0*(knee_targ[0] - s[6])  - 0.25*s[7]
        if knee_targ[1]: knee_todo[1] = 4.0*(knee_targ[1] - s[11]) - 0.25*s[12]

        hip_todo[0] -= 0.9*(0-s[0]) - 1.5*s[1] # PID to keep head strait
        hip_todo[1] -= 0.9*(0-s[0]) - 1.5*s[1]
        knee_todo[0] -= 15.0*s[3]  # vertical speed, to damp oscillations
        knee_todo[1] -= 15.0*s[3]

        a[0] = hip_todo[0]
        a[1] = knee_todo[0]
        a[2] = hip_todo[1]
        a[3] = knee_todo[1]
        a = np.clip(0.5*a, -1.0, 1.0)

        env.render()
        if done: break
