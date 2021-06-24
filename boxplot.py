
import matplotlib.pyplot as plt

num_seed_warn_quant = [
    [21, 20, 26, 20, 29, 14, 15, 28],  # good, looka, quantitative
    [71, 81, 72, 62, 88, 68, 84, 84],  # bad, looka, quantitative
    [7, 8, 8, 15, 11, 8, 3, 7],        # good, meta, quantitative
    [44, 32, 34, 37, 35, 34, 36, 37]   # bad, meta, quantitative
]

num_tot_warn_quant = [
    [184, 133, 138, 54, 276, 116, 58, 140],  # good, looka, quantitative
    [1632, 2021, 1931, 1712, 2333, 1537, 2229, 2022],  # bad, looka, quantitative
    [37, 76, 27, 215, 131, 85, 42, 35],        # good, meta, quantitative
    [758, 538, 641, 683, 721, 736, 600, 721]  # bad, meta, quantitative
]

num_seed_warn_qualit = [
    [5, 16, 0, 0, 0, 0, 0, 0] , # good, looka, qualitative
    [6, 56, 0, 59, 20, 3, 51, 55], # bad, looka, qualitative
    [4, 0, 0, 4, 0, 11, 0, 2],  # good, meta, qualitative
    [7, 9, 4, 10, 16, 10, 19, 13]   # bad, meta, qualitative
]

num_tot_warn_qualit = [
    [226, 816, 0, 0, 0, 0, 0, 0],   # good, looka, qualitative
    [291, 2242, 0, 2061, 928, 137, 1648, 2168],   # bad, looka, qualitative
    [17, 0, 0, 22, 0, 67, 0, 4],  # good, meta, qualitative
    [13, 27, 35, 129, 136, 35, 130, 128]   # bad, meta, qualitative
]

green_diamond = dict(markerfacecolor="g", marker="D")
fig, ax = plt.subplots()
ax.set_title("Num. Tot. Warn / Quantitative")
ax.set_ylabel("# Warnings")
ax.set_xlabel("Variant")
ax.boxplot(num_tot_warn_quant, flierprops=green_diamond)
ax.set_xticklabels(["LookA/Good", "LookA/Bad", "MetaM/Good", "MetaM/Bad"])
# plt.show()
plt.savefig("num_tot_quantitative.pdf")

# good, looka, qualitative over 66 seeds
# [5, 16, 0, 0, 0, 0, 0, 0]
# [226, 816, 0, 0, 0, 0, 0, 0]

# bad, looka, qualitative over 69 seeds
# [6, 56, 0, 59, 20, 3, 51, 55]
# [291, 2242, 0, 2061, 928, 137, 1648, 2168]

# bad, meta, qualitative over 69 seeds
# [7, 9, 4, 10, 16, 10, 19, 13]
# [13, 27, 35, 129, 136, 35, 130, 128]

# good, meta, qualitative over 66 seeds
# [4, 0, 0, 4, 0, 11, 0, 2]
# [17, 0, 0, 22, 0, 67, 0, 4]

# bad, meta, quantitative over 69 seeds
# [44, 32, 34, 37, 35, 34, 36, 37]
# [758, 538, 641, 683, 721, 736, 600, 721]

# good, meta, quantitative over 66 seeds
# [7, 8, 8, 15, 11, 8, 3, 7]
# [37, 76, 27, 215, 131, 85, 42, 35]

# good, looka, quantitative over 66 seeds
# [21, 20, 26, 20, 29, 14, 15, 28]  -
# [184, 133, 138, 54, 276, 116, 58, 140]

# bad, looka, quantitative over 69 seeds
# [71, 81, 72, 62, 88, 68, 84, 84]  -
# [1632, 2021, 1931, 1712, 2333, 1537, 2229, 2022]