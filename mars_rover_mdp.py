#STATE SPACE
min_f_by_np = {
    (5, "S"): 0,
    (4, "S"): 1, (4, "F"): 3,
    (3, "S"): 2, (3, "F"): 4,
    (2, "S"): 3, (2, "F"): 5,
    (1, "S"): 4, (1, "F"): 6,
    (0, "S"): 5, (0, "F"): 7,
}

max_f_by_np = {
    (5, "S"): 0,
    (4, "S"): 2, (4, "F"): 4,
    (3, "S"): 6, (3, "F"): 9,
    (2, "S"): 9, (2, "F"): 9,
    (1, "S"): 9, (1, "F"): 9,
    (0, "S"): 9, (0, "F"): 9,
}

def is_valid_state(f, n, p):
    if not (0 <= f <= 9 and 0 <= n <= 5 and p in ["S", "F"]):
        return False
    if p == "F" and n == 5:
        return False
    min_f = min_f_by_np.get((n, p), 0)
    max_f = max_f_by_np.get((n, p), 9)
    if f < min_f or f > max_f:
        return False
    if n == 3 and p == "F" and f == 7:
        return False
    return True
# Build all valid states using nested loops 
all_states = []
for f in range(10):
    for n in range(6):
        for p in ["S", "F"]:
            if is_valid_state(f, n, p):
                all_states.append((f, n, p))

print("Total valid states: " + str(len(all_states)))

# TRANSITION FUNCTION

def get_transitions(f, n, p, action):
    if n == 0:
        return []

    n_next = n - 1
    results = []

    if action == 'safe':
        p_next = "S"
        for fatigue, prob in [(1, 0.8), (2, 0.2)]:
            f_next = f + fatigue
            reward = 10
            if n == 1:
                reward += 25
            if f_next >= 10:
                reward = -10 * n
                results.append((prob, 'FAIL', reward))
            else:
                results.append((prob, (f_next, n_next, p_next), reward))

    elif action == 'fast':
        p_next = "F"
        if p == "S":
            base_increments = [(3, 0.7), (4, 0.3)]
        else:
            base_increments = [(5, 0.6), (7, 0.4)]

        micro_tear_active = (f >= 8)

        for fatigue, prob in base_increments:
            if micro_tear_active:
                micro_cases = [(0, 0.8), (4, 0.2)]
            else:
                micro_cases = [(0, 1.0)]

            for micro_fatigue, micro_prob in micro_cases:
                total_fatigue = fatigue + micro_fatigue
                joint_prob = prob * micro_prob
                f_next = f + total_fatigue

                reward = 10 + 3
                if n == 1:
                    reward += 25
                if f_next >= 10:
                    reward = -10 * n
                    results.append((joint_prob, 'FAIL', reward))
                else:
                    results.append((joint_prob, (f_next, n_next, p_next), reward))
    return results

# DYNAMIC PROGRAMMING (BACKWARD INDUCTION)


V = {}
Policy = {}

# t=0: terminal states, value = 0
V[0] = {}
for (f, n, p) in all_states:
    if n == 0:
        V[0][(f, n, p)] = 0.0


# Backward induction t=1 to t=5
for t in range(1, 6):
    V[t] = {}
    Policy[t] = {}

    for (f, n, p) in all_states:
        if n != t:
            continue

        best_value = None
        best_action = None

        for action in ['safe', 'fast']:
            transitions = get_transitions(f, n, p, action)
            if not transitions:
                continue

            expected = 0.0
            for prob, s_next, r in transitions:
                if s_next == 'FAIL':
                    future = 0.0
                else:
                    f2, n2, p2 = s_next
                    future = V[t-1].get((f2, n2, p2), 0.0)
                expected += prob * (r + future)

            if best_value is None or expected > best_value:
                best_value = expected
                best_action = action

        if best_value is not None:
            V[t][(f, n, p)] = best_value
            Policy[t][(f, n, p)] = best_action

    for s in sorted(V[t].keys()):
        state_str = "(f=" + str(s[0]) + ", n=" + str(s[1]) + ", p=" + str(s[2]) + ")"
        val_str = str(round(V[t][s], 3))
        act_str = Policy[t].get(s, 'N/A')
        print("  " + state_str.ljust(25) + "  " + val_str.rjust(8) + "  " + act_str)

# 4. INITIAL STATE RESULT
print("OPTIMAL VALUE FROM INITIAL STATE (0, 5, S):")
initial = (0, 5, "S")
print("  V*(f=0, n=5, p=S) = " + str(round(V[5].get(initial, 0), 3)))
