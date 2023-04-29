'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np

epsilon = 1e-3

def compute_transition_matrix(model):
    '''
    Parameters:
    model - the MDP model returned by load_MDP()

    Output:
    P - An M x N x 4 x M x N numpy array. P[r, c, a, r', c'] is the probability that the agent will move from cell (r, c) to (r', c') if it takes action a, where a is 0 (left), 1 (up), 2 (right), or 3 (down).
    '''
import numpy as np

def compute_transition_matrix(model):
    M, N = model.M, model.N
    P = np.zeros((M, N, 4, M, N))

    directions = [(0, -1), (-1, 0), (0, 1), (1, 0)]

    for r in range(M):
        for c in range(N):
            if model.T[r, c]:  # Skip terminal states
                P[r,c,:,:,:] = 0
                continue
            for a, (dr, dc) in enumerate(directions):
                intended_r, intended_c = r + dr, c + dc

                # Check if the intended move is valid
                if 0 <= intended_r < M and 0 <= intended_c < N and not model.W[intended_r, intended_c]:
                    P[r, c, a, intended_r, intended_c] = model.D[r, c, 0]
                else:
                    P[r, c, a, r, c] += model.D[r, c, 0]

                # Check if the counter-clockwise move is valid
                ccw_r, ccw_c = r - dc, c + dr
                if 0 <= ccw_r < M and 0 <= ccw_c < N and not model.W[ccw_r, ccw_c]:
                    P[r, c, a, ccw_r, ccw_c] = model.D[r, c, 1]
                else:
                    P[r, c, a, r, c] += model.D[r, c, 1]

                # Check if the clockwise move is valid
                cw_r, cw_c = r + dc, c - dr
                if 0 <= cw_r < M and 0 <= cw_c < N and not model.W[cw_r, cw_c]:
                    P[r, c, a, cw_r, cw_c] = model.D[r, c, 2]
                else:
                    P[r, c, a, r, c] += model.D[r, c, 2]

    return P

def update_utility(model, P, U_current):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()
    P - The precomputed transition matrix returned by compute_transition_matrix()
    U_current - The current utility function, which is an M x N array

    Output:
    U_next - The updated utility function, which is an M x N array
    '''
    M, N = model.M, model.N
    gamma = model.gamma
    R = model.R
    U_next = np.zeros((M, N))
    #initialize the terminal utility again
    terminal_indices = np.argwhere(model.T)
    for terminal_indice in terminal_indices:    
        U_next[terminal_indice[0],terminal_indice[1]]=model.R[terminal_indice[0],terminal_indice[1]]
    for r in range(M):
        for c in range(N):
            if not model.T[r, c]:
                action_values = []
                for a in range(4):  # Iterate through the 4 actions (left, up, right, down)
                    expected_reward = sum([P[r, c, a, r_next, c_next] * U_current[r_next, c_next] for r_next in range(M) for c_next in range(N)])
                    action_values.append(R[r, c] + gamma * expected_reward)
                U_next[r, c] = max(action_values)
    return U_next
def value_iteration(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()

    Output:
    U - The utility function, which is an M x N array
    '''
    max_iterations = 100
    M, N = model.M, model.N
    U_current = np.zeros((M, N))
    U_next = np.zeros((M, N))
    
    #initialize the terminal utility value
    terminal_indices = np.argwhere(model.T)
    for terminal_indice in terminal_indices:    
        U_current[terminal_indice[0],terminal_indice[1]]=model.R[terminal_indice[0],terminal_indice[1]]
    U_next = np.copy(U_current)
    
    P = compute_transition_matrix(model)

    for _ in range(max_iterations):
        U_next = update_utility(model, P, U_current)

        if np.max(np.abs(U_next - U_current)) < epsilon:
            break

        U_current = np.copy(U_next)

    return U_next
if __name__ == "__main__":
    import utils
    model = utils.load_MDP('models/small.json')
    model.visualize()
    U = value_iteration(model)
    model.visualize(U)
