import numpy as np

for i in range(1, 61):
    # A_0_cp = np.load(f"/home/yakir/debug/crypt/A_0_{i}.npy")
    # B_0_cp = np.load(f"/home/yakir/debug/crypt/B_0_{i}.npy")
    C_0_cp = np.load(f"/home/yakir/debug/crypt/C_0_{i}.npy")

    # A_1_cp = np.load(f"/home/yakir/debug/crypt/A_1_{i}.npy")
    # B_1_cp = np.load(f"/home/yakir/debug/crypt/B_1_{i}.npy")
    # C_1_cp = np.load(f"/home/yakir/debug/crypt/C_1_{i}.npy")

    # A_cl = np.load(f"/home/yakir/debug/client/A_{i}.npy")
    # B_cl = np.load(f"/home/yakir/debug/client/B_{i}.npy")
    C_cl = np.load(f"/home/yakir/debug/client/C_{i}.npy")

    # A_sr = np.load(f"/home/yakir/debug/server/A_{i}.npy")
    # B_sr = np.load(f"/home/yakir/debug/server/B_{i}.npy")
    # C_sr = np.load(f"/home/yakir/debug/server/C_{i}.npy")

    # assert np.all(A_0_cp == A_cl), i
    # assert np.all(B_0_cp == B_cl), i
    if not np.all(C_0_cp == C_cl):
        print(i % 20)
        print("=====")
    # assert np.all(C_0_cp == C_cl), i
    # assert np.all(C_1_cp == C_sr), i
    # assert np.all(A_1_cp == A_sr), i
    # assert np.all(B_1_cp == B_sr), i
