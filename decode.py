def get_sample_solution(name_to_value):#,node_n,feature_n):
    """ Returns all the possible solutions, or an empty tuple if no solution is found."""
    v_var = {}
    l_var = {}
    r_var = {}
    a_var = {}
    cl_var = {}
    cr_var = {}
    vl_var = []
    vr_var = []

    for k, v in name_to_value.items():
        try:
            if k.startswith('VL') and v == 1:
                vl_var.append(int(k[2:]))
            elif k.startswith('VR') and v == 1:
                vr_var.append(int(k[2:]))
            elif k.startswith('a') and v == 1:
                feature = k[1:].partition(',')[0]
                node = k[1:].partition(',')[2]
                a_var[int(node)] = int(feature)
            elif k.startswith('cl'):
                cl_var[int(k[2:])] = 1 if v == 1 else 0
            elif k.startswith('cr'):
                cr_var[int(k[2:])] = 1 if v == 1 else 0           

        except Exception as e:
            traceback.print_exc()
            raise e
    
    v_var = {}
    for i in range(1,len(cl_var)+1):
        if i in vl_var and i in vr_var:
            v_var[i] = 'A'
        if i in vl_var and i not in vr_var:
            v_var[i] = 'B'
        if i not in vl_var and i in vr_var:
            v_var[i] = 'C'
        if i not in vl_var and i not in vr_var:
            v_var[i] = 'D'
    
    
    solution = {'v': v_var, 'a': a_var, 'cl': cl_var, 'cr': cr_var}

    return solution