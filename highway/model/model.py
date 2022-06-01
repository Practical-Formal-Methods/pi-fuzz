import copy
import sys

import numpy as np
import itertools
from matplotlib import colors
from PIL import Image, ImageDraw

sys.path.append("highway/model")
# Project specific inputs
import constants as c
from drivers import Grandma, Speed_Maniac, Agent

# Colors
white = colors.to_hex("w")
black = colors.to_hex("k")
red = colors.to_hex("r")
cyan = colors.to_hex("c")
blue = colors.to_hex("b")
magenta = colors.to_hex("m")
yellow = colors.to_hex("y")
grey = (196, 196, 196)
green = colors.to_hex("g")
start_color = (76, 0, 153)
neon_green = (128, 255, 0)

# todo catch ratio = 1
# todo new car positions via constant file -> speeds


class Highway:
    def __init__(
        self,
        num_lines,
        length_lines,
        rng,
        mode="line_ratio",
        ratios=None,
        random_start=False,
        input_stripe=False,
    ):
        # possible modes: ['line_ratio', 'street_ratio', constant_distance']
        # features: [random_start, input_stripe]
        if ratios is None:
            ratios = []
        if mode in ["line_ratio", "constant_distance"] and len(ratios) != num_lines:
            print("You have to specify exact as many ratios as there are lines")
            return
        if mode == "street_ratio" and len(ratios) != 1:
            print("You have to specify one overall ratio")
            return

        # save the given simulation parameters
        self.num_lines = num_lines
        self.length_lines = length_lines
        self.mode = mode
        self.ratios = ratios
        self.ratio = ratios[0]
        self.random_start = random_start
        self.input_stripe = input_stripe
        self.num_steps = 0
        self.acc_return = 0
        self.rng = rng
        self.action_space = [0, 1, 2, 3, 4]

        # initialize street
        self.street = np.zeros((num_lines, length_lines)).tolist()

        # pre-processing of need numbers to run the simulation
        if mode == "street_ratio":
            self.cars_per_line_overall = int(self.ratio * self.length_lines)
            self.total_number_cars = int(self.ratio * num_lines * length_lines)

        if mode in ["line_ratio", "constant_distance"]:
            self.cars_per_lines = []
            self.dists = []
            for ratio in self.ratios:
                cpl = int(ratio * length_lines)
                self.cars_per_lines.append(cpl)
                # constant distance -> the space between cars is equally spread
                free_spots = self.length_lines - ratio
                # having n cars, there are n+1 distant spots
                # this can sometimes have the effect of having exactly n+1 cars, but the constant distance
                # is kept, which is here considered more important
                self.dists.append(int(free_spots / cpl - 0.5))

        # fill street initially
        if mode == "line_ratio":
            # non constant distance -> randomly spread cars per line
            for j, (line, ratio) in enumerate(zip(self.street, self.cars_per_lines)):
                spots = rng.choice(range(self.length_lines), ratio, replace=False)
                spots.sort()
                for i in range(self.length_lines):
                    if i in spots:
                        line[i] = self.get_new_car(j)
                    else:
                        line[i] = None

        if mode == "constant_distance":
            for j, (line, dist) in enumerate(zip(self.street, self.dists)):
                for i in range(self.length_lines):
                    if i % (dist + 1) == 0:
                        line[i] = self.get_new_car(j)
                    else:
                        line[i] = None

        if mode == "street_ratio":
            # ratio not given for lines but for whole street
            possible_spots = np.array(
                list(itertools.product(range(self.num_lines), range(self.length_lines)))
            )
            indexes = rng.choice(
                range(len(possible_spots)), self.total_number_cars, replace=False
            )
            spots = possible_spots[indexes].tolist()
            for j, line in enumerate(self.street):
                for i in range(self.length_lines):
                    if [j, i] in spots:
                        line[i] = self.get_new_car(j)
                    else:
                        line[i] = None

        # independent from the mode, set the car to the "last" line
        last_line = self.street[-1]
        if not random_start:
            # set the car to the beginning of the line
            if input_stripe:
                index = c.INPUT_STRIPE + 1
            else:
                index = 0
            while index < len(last_line):
                spot = last_line[index]
                if spot == None:
                    self.street[-1][index] = Agent()
                    break
                index += 1
            self.car_agent = Agent()
        else:
            # set the agent on a random spot on the street
            spot = True
            while spot != None:
                if input_stripe:
                    index = self.rng.integers(c.INPUT_STRIPE + 1, self.length_lines)
                else:
                    index = self.rng.integers(self.length_lines)
                spot = last_line[index]
            self.car_agent = Agent()
            self.street[-1][index] = self.car_agent

    def calculate_autoplayer_trajectory(self, line, spot, car):
        steps = []
        times = []
        out = False
        num_spots = car.get_v() + 1
        time_per_step = 1 / num_spots
        for i in range(num_spots):
            new_spot = spot + i
            steps.append((line, new_spot))
            times.append((i * time_per_step, (i + 1) * time_per_step))
            if new_spot >= self.length_lines:
                out = True
        return steps, times, out

    def calculate_agent_trajectory(self, line, spot, car, action):
        acc = 0
        change_line = 0

        # speeding
        # if action < 3:
        if action == c.SPEED_UP:
            acc = 1
        elif action == c.SLOW_DOWN:
            acc = -1
        # line change
        # else:
        if action == c.CHANGE_LEFT:
            change_line = -1
        elif action == c.CHANGE_RIGHT:
            change_line = 1

        # calculate new velocity car
        car.accelerate(acc)

        # if there is no line change, the calculation of trajectory does not differ from normal car
        # so far, the car did not break out of the lines -> crash = False
        if change_line == 0:
            steps, times, out = self.calculate_autoplayer_trajectory(line, spot, car)
            crash = False
        else:

            new_line = line + change_line
            if 0 <= new_line < self.num_lines:
                crash = False
            else:
                crash = True
            steps = []
            times = []
            out = False
            length_trajectory = car.get_v() + 1
            time_per_step = 1 / length_trajectory
            for i in range(length_trajectory):
                current_spot = spot + i
                # >= important if car only has speed 1 -> wont change line otherwisei
                if i >= length_trajectory / 2:
                    current_line = new_line
                else:
                    current_line = line

                steps.append((current_line, current_spot))
                times.append((i * time_per_step, (i + 1) * time_per_step))

                if (
                    current_spot >= self.length_lines
                    or current_line < 0
                    or current_line >= self.num_lines
                ):
                    out = True

        return steps, times, out, crash

    def step(self, action):
        if action not in c.AVAILABLE_ACTIONS:
            print("invalid action")
            return None, None, None
        # save all trajectories and number of removed cars
        trajectories = []
        trajectories_times = []
        removed_cars = np.zeros((self.num_lines))
        agent_trajectory = []
        agent_times = []
        agent_out = False
        agent_crash = False

        # walk the streets backwards, as otherwise we could delete cars
        for i in range(self.length_lines)[::-1]:
            for j in range(self.num_lines)[::-1]:
                # take current car
                car = self.street[j][i]

                # if there is no car, continue
                if car == None:
                    continue

                # if car is no agent, it is an autoplayer
                if car.kind() != "A":
                    trajectory, times, out = self.calculate_autoplayer_trajectory(
                        j, i, car
                    )
                    # save trajectory
                    trajectories.append(trajectory)
                    trajectories_times.append(times)
                    # remove car from old position
                    self.street[j][i] = None
                    # if car is not finished, i.e. left the length of the street
                    # that we observe, set it to its new position
                    # otherwise remember, cause we need to add new cars
                    if not out:
                        next_pos = trajectory[-1]
                        self.street[next_pos[0]][next_pos[1]] = car
                    else:
                        removed_cars[j] += 1
                # this is an agent, so it is possible that there are crashes with walls or other cars
                else:
                    # save the traj, out and crash for later
                    (
                        agent_trajectory,
                        agent_times,
                        agent_out,
                        agent_crash,
                    ) = self.calculate_agent_trajectory(j, i, car, action)
                    self.street[j][i] = None

                    if (not agent_out) and (not agent_crash):
                        next_pos = agent_trajectory[-1]
                        self.street[next_pos[0]][next_pos[1]] = car

        car_crashed = agent_crash
        # check for all positions & times of agent:
        for a_pos, a_t in zip(agent_trajectory, agent_times):
            if car_crashed:
                # if car has already crashed, there is no need to continue
                break
            # compare with all other trajectories
            for trajectory, times in zip(trajectories, trajectories_times):
                # if the trajectories intersect
                if a_pos in trajectory:
                    index = trajectory.index(a_pos)
                    time = times[index]
                    other_s_time, other_e_time = time[0], time[1]
                    agent_s_time, agent_e_time = a_t[0], a_t[1]
                    # and also the times intersect!
                    if other_s_time > agent_e_time or agent_s_time > other_e_time:
                        car_crashed = False
                    else:
                        car_crashed = True
                        break
                    # if a_t[1] < starting_time < a_t[1]:
                    #     # the cars have crashed
                    #     car_crashed = True
                    #     # print("cars crashed at: ", a_pos, a_t, time)
                    #     break

        # insert new cars
        if self.mode in ["line_ratio", "constant_distance"]:
            for j, (line, dist) in enumerate(zip(self.street, self.dists)):
                first = self.length_lines - 1
                for i, spot in enumerate(line):
                    if spot != None:
                        first = i
                        break
                # mode constant distance
                if self.mode == "constant_distance":
                    spot = first - dist - 1
                    while spot >= 0:
                        line[spot] = self.get_new_car(j)
                        spot = spot - dist - 1
                # mode line ratio
                else:
                    for _ in range(int(removed_cars[j])):
                        index = self.rng.integers(self.length_lines)
                        while line[index] != None:
                            index = self.rng.integers(self.length_lines)

                        line[index] = self.get_new_car(j)

        if self.mode == "street_ratio":
            # not equally spread
            number_new_cars = int(np.sum(removed_cars))
            for _ in range(number_new_cars):
                index_a, index_b = (
                    self.rng.integers(self.num_lines),
                    self.rng.integers(self.length_lines),
                )
                # I assume sparse traffic, so this is more cheap than calculating the candidates
                while self.street[index_a][index_b] != None:
                    index_a, index_b = (
                        self.rng.integers(self.num_lines),
                        self.rng.integers(self.length_lines),
                    )

                self.street[index_a][index_b] = self.get_new_car(index_a)

        # return reward and next state
        state_nn, _, _ = self.get_state(
            one_hot=True, linearize=True, window=True, distance=True
        )

        if car_crashed:
            reward = c.LOOSE_REWARD
            done = True
        else:
            if agent_out:
                reward = c.WIN_REWARD
                done = True
            else:
                reward = c.STEP_REWARD
                done = False

        # save the acc return
        self.num_steps += 1
        # self.acc_return += np.power(reward, c.GAMMA)   ## THIS IS CORRECTED BELOW
        self.acc_return += np.power(c.GAMMA, self.num_steps) * reward

        return reward, state_nn, done

    def set_state(self, hi_lvl_state, rng):
        street, v = hi_lvl_state
        # self.reset(None)
        self.rng = rng
        self.street = copy.deepcopy(street)
        self.set_v(v)
        self.num_steps = 0
        self.acc_return = 0

    def get_car_pos(self):
        car_pos = []
        for lid, lane in enumerate(self.street):
            for pid, p in enumerate(lane):
                if p is not None:
                    car_pos.append((lid, pid))

        return car_pos


    def get_state(
        self,
        one_hot=False,
        linearize=False,
        window=False,
        distance=False,
        velocity=True,
    ):
        v = self.get_v()
        pos = None
        if window and not self.input_stripe:
            print(
                "window state is only possible if you make use of an striped environment"
            )

        if one_hot:
            occ = np.zeros((self.num_lines, self.length_lines))
            grandmas = np.zeros((self.num_lines, self.length_lines))
            maniacs = np.zeros((self.num_lines, self.length_lines))
            agent = np.zeros((self.num_lines, self.length_lines))
            for j, line in enumerate(self.street):
                for i, car in enumerate(line):
                    if car != None:
                        occ[j][i] = 1
                        if car.kind() == "G":
                            grandmas[j][i] = 1
                        if car.kind() == "S":
                            maniacs[j][i] = 1
                        if car.kind() == "A":
                            pos = i
                            agent[j][i] = 1

            if window:
                if pos != None:
                    f = pos - c.WINDOW_STRIPE
                    t = pos + c.WINDOW_STRIPE
                else:
                    f = 0
                    t = 0

                if t > self.length_lines - 1:
                    diff = t - (self.length_lines - 1)
                    t = self.length_lines - 1

                w_occ = np.zeros((self.num_lines, 2 * c.WINDOW_STRIPE + 1))
                w_grandmas = np.zeros((self.num_lines, 2 * c.WINDOW_STRIPE + 1))
                w_maniacs = np.zeros((self.num_lines, 2 * c.WINDOW_STRIPE + 1))
                w_agent = np.zeros((self.num_lines, 2 * c.WINDOW_STRIPE + 1))

                w_occ[:, f - f : (t - f)] = occ[:, f:t]
                w_grandmas[:, f - f : t - f] = grandmas[:, f:t]
                w_maniacs[:, f - f : t - f] = maniacs[:, f:t]
                w_agent[:, f - f : t - f] = agent[:, f:t]

                occ = w_occ
                grandmas = w_grandmas
                maniacs = w_maniacs
                agent = w_agent

            if pos == None:
                distance_feature = 0
            else:
                distance_feature = self.length_lines - 1 - pos

            if linearize:
                if window:
                    reshape_size = self.num_lines * (c.WINDOW_STRIPE * 2 + 1)
                else:
                    reshape_size = self.num_lines * self.length_lines
                occ = occ.reshape(reshape_size)
                grandmas = grandmas.reshape(reshape_size)
                maniacs = maniacs.reshape(reshape_size)
                agent = agent.reshape(reshape_size)

                if distance and velocity:  # both
                    return np.concatenate(
                        [occ, grandmas, maniacs, agent, [distance_feature, v]]
                    ), copy.deepcopy(self.street), self.rng
                else:
                    if distance:  # only distance
                        return np.concatenate(
                            [occ, grandmas, maniacs, agent, [distance_feature]]
                        ), copy.deepcopy(self.street), self.rng
                    else:
                        if velocity:  # only velocity
                            return np.concatenate(
                                [occ, grandmas, maniacs, agent, [v]]
                            ), copy.deepcopy(self.street), self.rng
                        else:  # None
                            return np.concatenate(
                                [occ, grandmas, maniacs, agent]
                            ), copy.deepcopy(self.street), self.rng

            else:
                if distance and velocity:
                    return occ, grandmas, maniacs, agent, distance_feature, v
                if distance:
                    return occ, grandmas, maniacs, agent, distance_feature
                if velocity:
                    return occ, grandmas, maniacs, agent, v
                # none of the former options
                return occ, grandmas, maniacs, agent
        else:
            if linearize or window or distance:
                print(
                    "linearizing, window or distance without one hot is not supported yet."
                )

            return self.street

    def get_new_car(self, line_id):
        if line_id < int(self.num_lines - 0.5):
            return Speed_Maniac()
        else:
            return Grandma()

    def reset(self, rng=None):
        if rng is None:
            rng = self.rng
        self.__init__(
            self.num_lines,
            self.length_lines,
            rng=rng,
            mode=self.mode,
            ratios=self.ratios,
            random_start=self.random_start,
            input_stripe=self.input_stripe,
        )

    def show(self):
        w, h = self.length_lines, self.num_lines
        size = 100
        t_w, t_h = size * w, size * h

        gap = 5
        middle = int(t_h / 2)

        result = Image.new("RGB", (t_w, t_h), grey)
        draw = ImageDraw.Draw(result)
        draw.line((gap, gap, t_w - gap, gap), fill=white, width=2)
        draw.line((gap, t_h - gap, t_w - gap, t_h - gap), fill=white, width=2)
        for i in range(self.length_lines):
            draw.line(
                (i * size + 2 * gap, middle, (i + 1) * size - 2 * gap, middle),
                fill=white,
                width=2,
            )

        s = Image.open("highway/cars/blue.png").convert("RGBA")
        s = s.resize((size - 2 * gap, int(s.size[1] * (size - 2 * gap) / s.size[0])))
        g = Image.open("highway/cars/green.png").convert("RGBA")
        g = g.resize((size - 2 * gap, int(g.size[1] * (size - 2 * gap) / g.size[0])))
        a = Image.open("highway/cars/red.png").convert("RGBA")
        a = a.resize((size - 2 * gap, int(a.size[1] * (size - 2 * gap) / a.size[0])))

        for j, line in enumerate(self.street):
            for i, car in enumerate(line):
                if car == None:
                    continue
                if car.kind() == "G":
                    result.paste(g, (i * size + gap, j * size + gap + 20), mask=g)
                    continue
                if car.kind() == "S":
                    result.paste(s, (i * size + gap, j * size + gap + 20), mask=s)
                if car.kind() == "A":
                    result.paste(a, (i * size + gap, j * size + gap + 20), mask=a)

        return result

    def get_v(self):
        # hand request to the agent
        return self.car_agent.get_v()

    def set_v(self, v):
        # hand request to the agent
        return self.car_agent.set_v(v)

    def get_discounted_return(self):
        return self.acc_return

    def applicable_actions(self):
        return self.action_space
