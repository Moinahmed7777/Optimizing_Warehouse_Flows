
"""
Created on Wed May 27 16:42:28 2020

"""
#Optimizing Warehouse Flows with Q-learning

import numpy as np


#discount factor, gamma and Learning rate, alpha
gamma= 0.75
alpha= 0.9


#states
location_to_state= {"A" : 0,
                    "B" : 1,
                    "C" : 2,
                    "D" : 3,
                    "E" : 4,
                    "F" : 5,
                    "G" : 6,
                    "H" : 7,
                    "I" : 8,
                    "J" : 9,
                    "K" : 10,
                    "L" : 11}
key_list = list(location_to_state.keys()) 
val_list = list(location_to_state.values())


actions=[0,1,2,3,4,5,6,7,8,9,10,11]

#rewards
R = np.array([[0,1,0,0,0,0,0,0,0,0,0,0],
              [1,0,1,0,0,1,0,0,0,0,0,0],
              [0,1,0,0,0,0,1,0,0,0,0,0],
              [0,0,0,0,0,0,0,1,0,0,0,0],
              [0,0,0,0,0,0,0,0,1,0,0,0],
              [0,1,0,0,0,0,0,0,0,1,0,0],
              [0,0,1,0,0,0,1,1,0,0,0,0],
              [0,0,0,1,0,0,1,0,0,0,0,1],
              [0,0,0,0,1,0,0,0,0,1,0,0],
              [0,0,0,0,0,1,0,0,1,0,1,0],
              [0,0,0,0,0,0,0,0,0,1,0,1],
              [0,0,0,0,0,0,0,1,0,0,1,0]])


#initialing q values

def Q_learning(Rc):
    Q= np.array(np.zeros([12,12]))
    new_R=np.copy(Rc)
    for i in range(1000):
        cur_state=np.random.randint(0, 12)
        
        emp=[]
        for j in range(len(new_R[cur_state])):
            if new_R[cur_state,j]>0:
                emp.append(j)
        next_state=np.random.choice(emp)
        
        TD = new_R[cur_state,next_state] + gamma*Q[next_state, np.argmax(Q[next_state,])] - Q[cur_state,next_state]
        
        Q[cur_state,next_state] += alpha *TD
        
    return Q.astype(int)




def shortest_path(start,Goal):
   
    Rc=np.copy(R)
    Rc[location_to_state[Goal],location_to_state[Goal]]=1000
   
    Q=Q_learning(Rc)
    
    get_start_index=location_to_state[start]
    
    
    index_path = []
    index_path.append(get_start_index)
    actual_path=[]
    actual_path.append(start)
    
    while True:
        gni=index_path[-1]
        next_index = np.argmax(Q[gni,])
        
        index_path.append(next_index)
        g= key_list[val_list.index(next_index)]
        actual_path.append(g)
        if Goal==g:
            
            break
    
    
    return actual_path

def best_path(s,i,g):
    return shortest_path(s,i) + shortest_path(i,g)[1:]
print(best_path("E","K","G"))

