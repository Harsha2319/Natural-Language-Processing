import sys

# This function hardcodes the Model tables as dictionaries
def tables():
    tags = ['NNP', 'MD', 'VB', 'JJ', 'NN', 'RB', 'DT']

    pie = dict(zip(tags, [0.2767, 0.0006, 0.0031, 0.0453, 0.0449, 0.0510, 0.2026]))

    a_tags = [(t1,t2) for t1 in tags for t2 in tags]
    a_values = [0.3777, 0.0110, 0.0009, 0.0084, 0.0584, 0.0090, 0.0025,
                0.0008, 0.0002, 0.7968, 0.0005, 0.0008, 0.1698, 0.0041,
                0.0322, 0.0005, 0.0050, 0.0837, 0.0615, 0.0514, 0.2231,
                0.0366, 0.0004, 0.0001, 0.0733, 0.4509, 0.0036, 0.0036,
                0.0096, 0.0176, 0.0014, 0.0086, 0.1216, 0.0177, 0.0068,
                0.0068, 0.0102, 0.1011, 0.1012, 0.0120, 0.0728, 0.0479,
                0.1147, 0.0021, 0.0002, 0.2157, 0.4744, 0.0102, 0.0017]

    a = dict(zip(a_tags, a_values))

    words = ['Janet', 'will', 'back', 'the', 'bill']
    b_tags = [(w,t) for t in tags for w in words]
    b_values = [0.000032, 0, 0, 0.000048, 0,
                0, 0.308431, 0, 0, 0,
                0, 0.000028, 0.000672, 0, 0.000028,
                0, 0, 0.000340, 0, 0,
                0, 0.000200, 0.000223, 0, 0.002337,
                0, 0, 0.010446, 0, 0,
                0, 0, 0, 0.506099, 0]

    b = dict(zip(b_tags, b_values))
    return tags, pie, a, b

# Returns the max probability and the sequence of tags for input sequence
def viterbi(states, pie, a, b, o):
    v = {}
    back = {}
    for s in states:
        v[s,1] = pie[s] * b[o[0], s]
        back[(s, 1)] = 0
    for t in range(2, len(o)+1):
        for s in states:
            v_list = []
            for prev_s in states:
                v_list.append(v[prev_s,t-1] * a[prev_s, s] * b[o[t-1], s])
            v[s, t] = max(v_list)
            back[s,t] = states[v_list.index(max(v_list))]

    v_final_list = []
    for s in states:
        v_final_list.append(v[s,len(o)])
    v_final = max(v_final_list)
    backtrack_path = []
    backtrack = states[v_final_list.index(v_final)]
    backtrack_path.append(backtrack)

    for t in range(len(o),0,-1):
        backtrack = back[backtrack, t]
        backtrack_path.append(backtrack)
        for s in states:
            if v[s, t] != 0:
                print(t, s, v[s, t], back[s, t])
    backtrack_path.remove(0)
    backtrack_path.reverse()
    return v_final, backtrack_path

def main():
    tags, pie, a, b = tables()
    observation = sys.argv[1]
    o = observation.split(' ')
    print(o)
    print()
    prob, tags = viterbi(tags, pie, a, b, o)
    print()
    print('Final results')
    print('Max Probability : ',prob)
    print('POS tags for the input sequence : ',tags)

main()