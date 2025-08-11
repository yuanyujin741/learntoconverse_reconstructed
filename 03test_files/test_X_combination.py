# for the generation of X_combination
def get_reduced_X_combination(N,K, debug = True):
    """
    get the reduced X_combination of N,K
    """
    calculate_X_combination = calculate_reduced_X_combination(N,K,False)
    if N == 2 and K == 2:
        X_combinations = ["X12"]
    elif N == 2 and K == 3:
        X_combinations = ["X112"]
    elif N == 3 and K == 2: # 32
        X_combinations = ["X12","X13"]
    elif N == 3 and K == 3: # 33
        X_combinations = ["X112","X113","X123"]
    elif N == 2 and K == 4: # 24
        X_combinations = ["X1112","X1122"]
    elif N == 4 and K == 2: # 42
        X_combinations = ["X12","X13","X14"]
    elif N == 4 and K == 3: #43
        X_combinations = ["X112","X113","X114","X123"]
    elif N == 4 and K == 4:
        X_combinations = ["X1112","X1123","X1234"]
    elif N == 4 and K == 5:
        X_combinations = ["X11112","X11123","X11234"]
    elif N == 5 and K == 4:
        X_combinations = ["X1112","X1123","X1234"]
    elif N == 5 and K == 6:
        X_combinations = ["X111112","X111123","X111234","X112345"]
    elif N == 5 and K == 5:
        X_combinations = ["X11112","X11123","X11234","X12345"]
    elif N == 6 and K == 4:
        X_combinations = ["X1112","X1123","X1234"]
    else:
        X_combinations = calculate_reduced_X_combination(N,K,debug)
    if (X_combinations != calculate_X_combination) and debug:
        print("X_combinations{} is not equal to calculate_X_combination{} for N,K={},{}".format(X_combinations,calculate_X_combination,N,K))
    return X_combinations

def calculate_reduced_X_combination(N,K,debug = True):
    """
    calculate the reduced X_combination of N,K
    """
    window = [1] * (K - 2)
    range_window = list(range(1, N+1))
    window.extend(range_window)
    if debug:
        print(window)
    format_str = "X"+"".join(["{}"]*K)
    range_num = min(len(window) - K +1, K - 2 + 1)
    X_combinations = [format_str.format(*window[i:i+K]) for i in range(range_num)]
    if debug:
        print(X_combinations)
    return X_combinations

for N,K in [(2,2),(2,3),(3,2),(3,3),(2,4),(4,2),(4,3),(4,4),(4,5),(5,4),(5,6),(5,5),(6,4)]:
    get_reduced_X_combination(N,K,True)
print(get_reduced_X_combination(6,6))
