import sys 
import numpy as np
import timeit


class HMMTagger(object):
    def __init__(self, A_init={}, B_init={}, pi_init={}, tags=[], words=[], obs_lst=[], mode='train'):
        '''
        This is initializer function for HMM tagger. 
        
        If mode is train then it accepts initial trasition probabilities, 
        initial emission probabilities and initial state distribution to initialize an HMM.
        
        IF mode is test then it loads parameters from presaved model and converts it into log space 
        to do viterbi decoding.
        '''
        self.mode = mode
        
        if self.mode == 'test':
            self.load()
            self.get_log_hmm()
        else:
            self.A = A_init # transition probabilities
            self.B = B_init # emission probabilities
            self.pi = pi_init ## initial state distribution
            self.states = tags # list of tags becomes states in HMM
            self.symbols = words # unique words become symbols(observation) in HMM 
            self.obs_lst_big =  obs_lst # Training data, contains list of sentences
            print('Number of observation sequences => {}'.format(len(obs_lst)))

    def load(self, file='trained_states.npz'):
        data = np.load(file)
        self.A = data['A'].item()
        self.B = data['B'].item()
        self.pi = data['pi'].item()
        self.states = self.A.keys()
        
    def save(self, file='trained_states.npz'):
        np.savez(file, A=self.A, B=self.B, pi=self.pi)
    def forward_pass(self, obs):
        '''
        Given the observation sequence obs, this function computes the alpha values at each
        time step for each states in the HMM.
        '''
        #print('Doing forward pass [*]')
        # fwd_lst is a list which stores computed alpha values at each time step.
        self.fwd_lst = [{}]
        #print(self.fwd_lst)
        # using initial probability distribution and first observation to initialize the 
        #boundary condition for alpha computation 
        #print('obs0', obs[0])
        
        for state in self.states:
            self.fwd_lst[0][state] = self.pi[state] * self.B[state].get(obs[0], 0.)
        '''    
        sum1 = 0.
        for state in self.states:
        
            if self.fwd_lst[0][state] != 0:
                print(state, self.fwd_lst[0][state])
                print('Ashutosh')
                sum1 += self.fwd_lst[0][state]
        print('c_now', 1./sum1)
        '''
        # c_lst stores the scaling parameter for forward pass as described in the paper by Rabiner.
        self.c_lst = []
        self.c_lst.append(self.get_c(self.fwd_lst[0]))
        #print('c_fun', self.c_lst[0])
        #sys.exit()
        # scaling alphas for boundary condition
        for state in self.states:
            self.fwd_lst[0][state] = self.fwd_lst[0][state] * self.c_lst[0]
        '''
        for state in self.states:
            if self.fwd_lst[0][state] != 0:
                print(state, self.fwd_lst[0][state])
                print('Ashutosh')
        sys.exit()
        '''        
           
        # forward loop for each word in the word sequence
        for t in range(1, len(obs)):
            self.fwd_lst.append({})
            alpha_t = {}
            for curr_state in self.states:
                sum1 = 0.
                for prev_state in self.states:
                    # summing over all the previous states
                    if self.A[prev_state].has_key(curr_state):
                        sum1 += self.fwd_lst[t-1][prev_state] * self.A[prev_state][curr_state] * \
                        self.B[curr_state].get(obs[t], 0)
                alpha_t[curr_state] = sum1
                
            # compute c for current observation
            c = self.get_c(alpha_t)
            self.c_lst.append(c)
            
            # Scale current alpha values
            for curr_state in self.states:
                self.fwd_lst[t][curr_state] = c * alpha_t[curr_state]
        
    def backward_pass(self, obs):
        '''
        Given the observation sequence obs, this function computes the beta values at each 
        time step for each states in theHMM.
        '''
        #print('Doing backward pass [*]')
        num_obs = len(obs)
        # bwd_lst is a list which stores computed beta values at each time step.
        self.bwd_lst = []
        for i in range(num_obs):
            self.bwd_lst.append({})
        
        
        # initializing boundary condition for backward pass with scaling
        for state in self.states:
            self.bwd_lst[num_obs-1][state] = 1.*self.c_lst[num_obs-1]
        '''
        for i in range(num_obs):
            print(self.bwd_lst[i])
            print('*'*10)
        '''
        
         # backward loop for each word in the word sequence
        for t in range(num_obs-1)[::-1]:
            beta_t = {}
            for curr_state in self.states:
                sum1=0
                for next_state in self.states:
                    if self.A[curr_state].has_key(next_state):
                        sum1 += self.bwd_lst[t+1][next_state]*self.A[curr_state][next_state]*\
                        self.B[next_state].get(obs[t+1], 0.)
                beta_t[curr_state] = sum1
                
            for curr_state in self.states:
                self.bwd_lst[t][curr_state] = self.c_lst[t] * beta_t[curr_state]
        '''
        for i in range(num_obs):
            print(self.bwd_lst[i])
            print('*'*10)
        '''
            
    def get_c(self, alpha):
        '''
        This function computes scale value given the alphas for all the states at a given time.
        Read Rabiner paper to understand this parameter.'
        '''
        #print('Computing scale parameters [*]')
        sum1 = 0.
        for state in self.states:
            sum1 += alpha[state]
        if sum1 == 0:
            c = 1
        else:
            c = 1. / sum1
        return c
    def train(self):
        '''
        This function learns the parameters of the HMM using observation sequences in self.obs_lst
        '''
        num_iter = 1
        start = 0
        N = len(self.states)
        batch_size = len(self.obs_lst_big)
        end = start + batch_size 
        for iteration in range(num_iter):
            start1 = timeit.default_timer()
            print(' {} Started training on batch => {} {}'.format('#'*20, iteration, '#'*20))
            self.obs_lst = self.obs_lst_big[start:end]
            self.get_gamma_epsilon_table()            
            temp_aij = {} # temprary transition probabilities
            temp_bjk = {} # temporary emission probabilities
            temp_pi = {} # temporary initial state probabilities

            for i, curr_state in enumerate(self.states):
                print('Working on tag {}, {}/{}'.format(curr_state, i, N))
                temp_aij[curr_state] = {}
                temp_bjk[curr_state] = {}
                temp_pi[curr_state] = self.get_pi(curr_state)
                for sym in self.B[curr_state].keys():
                    temp = self.get_bj(curr_state, sym)
                    temp_bjk[curr_state][sym] = temp
                    #if temp != 0:
                        #print('Non_Zero_bj', temp)
                    
                for next_state in self.states:
                    if self.A[curr_state].has_key(next_state):
                        temp_aij[curr_state][next_state] = self.get_aij(curr_state, next_state)
                        
            sum1 = 0.0
            for v in temp_pi.values():
                #if v == 0:
                    #v = 10e-7
                sum1 += v
            for k, v in temp_pi.items():
                temp_pi[k] = v / sum1

            self.A = temp_aij
            self.B = temp_bjk
            self.pi = temp_pi
            #start = start + batch_size
            #end = end + batch_size
            #if end > len(self.obs_lst_big):
                #start = 0
                #end = start + batch_size
            end1 = timeit.default_timer()
            self.get_log_hmm()
            _, path = self.viterbi_decoding(self.obs_lst_big[0])
            print('tag_ouput => {}'.format(path))
            print('*'*100)
            print(self.A)
            print('*'*100)
            #print(self.B)
            print('*'*100)
            print(self.pi)
            print('*'*100)
            #sys.exit()
            print('{} Finished training batch {}. time taken = {:.4f} {}'.format('#'*20, iteration, end1 - start1, '#'*20))
             
        
        return (temp_aij, temp_bjk, temp_pi)
    def get_gamma_epsilon_table(self):
        '''
        This function computes gamma and epsilon table using all the observation sequences. 
        '''
        print('Computing gamma epsilon table [*]')
        start = timeit.default_timer()
        gamma_table = []
        epsilon_table = []
        for i in range(len(self.obs_lst)):
            gamma_table.append([])
            epsilon_table.append([])
        print('num_obs', len(gamma_table))
        #sys.exit()
        for i, obs in enumerate(self.obs_lst):
            #print(obs)
            #print('*'*10)
            #print('')
            self.forward_pass(obs)
            self.backward_pass(obs)
            #print('alphas => {}'.format(self.fwd_lst[len(obs)-1]))
            #print('c_lst => {}'.format(self.c_lst[len(obs)-1]))
            #sys.exit()
            num_obs = len(obs)
            #if num_obs==1:
                #continue
            #print(i, num_obs)
            #print('*'*10)
            gamma_table_temp = []
            for k in range(num_obs):
                gamma_table_temp.append({})
            epsilon_table_temp = []
            for k in range(num_obs-1):
                epsilon_table_temp.append({})
            
            for t in range(num_obs):
                for curr_state in self.states:
                    gamma_table_temp[t][curr_state] = self.get_gamma(curr_state, t)
            for t in range(num_obs-1):
                for curr_state in self.states:
                    epsilon_table_temp[t][curr_state] = {}
                    for next_state in self.states:
                        if self.A[curr_state].has_key(next_state):
                            epsilon_table_temp[t][curr_state][next_state] = self.get_epsilon(curr_state, 
                                                                                             next_state, 
                                                                                             obs[t + 1], t)
                
            gamma_table[i] = gamma_table_temp
            epsilon_table[i] = epsilon_table_temp
        #print('gamma_table', gamma_table)
        #print('epsiolon_table',epsilon_table)
        #print('num_obs', len(gamma_table))
        #sys.exit()
        self.gamma_table = gamma_table
        self.epsilon_table = epsilon_table
        end = timeit.default_timer()
        print('Done computing gamma epsilon table. Time taken = {:.4f} seconds.'.format(end-start))
        #print(self.gamma_table)
            
        
    def get_gamma(self, state, t):
        '''
        This function computes the gamma variable defined in HMM training. 
        '''
        #print('Computing gamma [*]')
        g = (self.fwd_lst[t][state] * self.bwd_lst[t][state]) / self.c_lst[t]
        return g
    def get_epsilon(self, curr_state, next_state, symbol, t):
        '''
        This function computes value of epsilon variable defined in an HMM training. 
        Epsilon value is calculated for the given current state, next state ant time.
        It uses alpha value at time t, beta value at time t+1 and the current transition and emission probabilities.
        '''
        #print('Computing epsilon [*]')
        eps = self.fwd_lst[t][curr_state] * self.A[curr_state][next_state] * self.B[next_state].get(symbol, 0) * \
        self.bwd_lst[t+1][next_state] 
        return eps
    
    def get_pi(self, state):
        '''
        This function computes the next initial state probability of a given state using recently calculated gamma table. 
        '''
        #print('Computing new pi [*]')
        pi = 0.0
        for i in range(len(self.gamma_table) ): 
            pi += self.gamma_table[i][0][state] 
        return pi 
    
    def get_aij(self, curr_state, next_state):
        '''
        This function returns the value of next transition probability for the given current and next state.
        It uses recently calculated gamma and epsilon tables.
        '''
        #print('Computing new aij [*]')
        num = 0.0
        den = 0.0
        for i in range(len(self.epsilon_table)): 
            for t in range(len(self.epsilon_table[i])): 
                den += self.gamma_table[i][t][curr_state] 
                num += self.epsilon_table[i][t][curr_state][next_state] 
        #if den == 0:
            #aij = 0
        #else:
        if den == 0:
            #print('zero_aj_0', curr_state, next_state)
            #print('$'*1000)
            aij = 0
        else:
            aij = num / den
            
        return aij
    
    def get_bj(self, curr_state, symbol):
        '''
        This function computes the value of next emission probability for giben state and symbol.
        It uses recently calculated gamma table.
        '''
        #print('Computing new bj [*]')
        num =  0.0 
        den = 0.0
        for i in range(len(self.gamma_table)):
            for t in range(len(self.gamma_table[i])):
                den += self.gamma_table[i][t][curr_state] # counting for all symbols. This becomes denominator.
                if self.obs_lst[i][t] == symbol:
                    num += self.gamma_table[i][t][curr_state] # Counting only for input symobl. This becomes numerator. 
        #if den ==0:
            #bj = 0
        #else:
        
        if den ==0:
            bj = 0
            #print('zero_bj_0', curr_state, symbol)
            #print('#'*1000)
        else:
            bj = num / den
        
        return bj
    
    def get_log_hmm(self):
        '''
        This fuction transforms the HMM parameters into log space
        '''
        self.logA = {}
        self.logpi = {}
        self.logB = {}
        for curr_state in self.states:
            self.logA[curr_state] = {}
            self.logB[curr_state] = {}
            for next_state in self.A[curr_state].keys():
                if self.A[curr_state][next_state] == 0:
                    self.logA[curr_state][next_state] = -1*10e10
                else:
                     self.logA[curr_state][next_state] = np.log(self.A[curr_state][next_state])
            if self.pi[curr_state] == 0:
                self.logpi[curr_state] = -1*10e10
            else:
                self.logpi[curr_state] = np.log(self.pi[curr_state])
            for sym in self.B[curr_state].keys():
                if self.B[curr_state][sym] == 0:
                    self.logB[curr_state][sym] = -1*10e10
                else:
                    self.logB[curr_state][sym] = np.log(self.B[curr_state][sym])
                
                
    def viterbi_decoding(self, obs):
        '''
        This function uses viterbi algorith to compute the optimal tag sequence for a give word sequence
        '''
        max_var = []
        for i in range(len(obs)):
            max_var.append({}) # stores the maximum path prbability for each state till given time
        path = {}
        
        # Initialize boundary condition
        for init_state in self.states:
            max_var[0][init_state] = self.logpi[init_state] + self.logB[init_state].get(obs[0], -1*10e10)
            path[init_state] = [init_state]
        #print(max_var[0])
        #sys.exit()
        #print('max_var', max_var)
        #print('path', path)
        #sys.exit()
        # Viterbi iteration for t > 0
        for t in range(1, len(obs)):
            newpath = {}     
            for curr_state in self.states:
                max_temp = -1*np.inf
                #max_state = ['NN']
                #print('curr_state', curr_state)
                for prev_state in self.states:
                    prob = max_var[t-1][prev_state] + self.logA[prev_state].get(curr_state, -1*10e10) + \
                    self.logB[curr_state].get(obs[t], -1*10e10)
                    #print('prob', prob)
                    if prob >= max_temp:
                        max_temp = prob
                        #print('max_temp', max_temp)
                        #print('max_state', prev_state)
                        max_state = prev_state
                max_var[t][curr_state] = max_temp
                #print('max_state', max_state)
                newpath[curr_state] = path[max_state] + [curr_state]     
            path = newpath
        #print(max_var[len(obs)-1])
        #sys.exit()
        max_temp = -1*np.inf
        max_state = self.states[0]
        for state in self.states:
            prob = max_var[len(obs)-1].get(state, -1*np.inf)
            if prob >= max_temp:
                max_temp = prob
                max_state = state
        return prob, path[max_state]
        
            
    
        
        
            
            
            
    