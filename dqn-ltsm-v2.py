# DQN (Deep Q Network) algorithm implementation as described in
# Mnih et al. 'Playing Atari with Deep Reinforcement Learning', 2013
# https://arxiv.org/pdf/1312.5602.pdf
#
# Mauricio Arango, September, 2017. Based on code authored by Jaromir Janisch:
# https://jaromiru.com/2016/10/12/lets-make-a-dqn-debugging/
#
# This implementation can be used with multiple target problems. It interfaces
# with the target problems using the OpenAi API.
#
# It has been tested with the OpenAi CartPole and MountainCar simulators. The target
# simulator is specified by setting the PROBLEM global variable in the MAIN section of the code.
#


import random, numpy, math, gym



# -------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *


class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel()
        # self.model.load_weights("MountainCar-basic.h5")

    def _createModel(self):

        print(" brain stateCnt: ", self.stateCnt)

        # RNN model, LSTM or GRU.
        # Implemented so all the changes needed, resahping or inputs and outputs, is done
        # in the Brain class, leaving the rest of the code the same as compared when using
        # regular NN.

        model = Sequential()

        # Input shape: each input item has a sequence of RNN_NUM_STEPS time steps and each time step
        # has stateCnt dimensions.
        # The input tensor has dimensions rnn_batch_len x RNN_NUM_STEPS x stateCnt
        # Set return_sequences=True to produce one output per time step
        model.add(GRU(96, input_shape=(RNN_NUM_STEPS, self.stateCnt), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(Dense(output_dim=self.actionCnt, activation='linear'))

        #model.compile(loss='mse', optimizer='adam')

        model.compile(loss='mse',
                      optimizer=Adam(lr=LEARNING_RATE))


        return model

    def train(self, x, y, epoch=1, verbose=0):
        # Reshape input state tensor, x,  so that it has the RNN_NUM_STEPS time steps and each time step
        # has x.shape[1] = self.stateCnt dimensions
        x = numpy.reshape(x, (-1, RNN_NUM_STEPS, x.shape[1]))
        # Reshape the targets tensor, y,  so that it has the RNN_NUM_STEPS time steps and each time step
        # has y.shape[1] = self.actionCnt dimensions
        y = numpy.reshape(y, (-1, RNN_NUM_STEPS, y.shape[1]))
        #print("train x.shape: ", x.shape)
        #print("train y.shape: ", y.shape)
        #self.model.fit(x, y, batch_size=64, nb_epoch=epoch, verbose=verbose)
        self.model.fit(x, y, RNN_BATCH_SIZE, nb_epoch=epoch, verbose=verbose)

    def predict(self, s):
        #return self.model.predict(s)
        #print("predict s.shape: ", s.shape)
        return self.model.predict(s.reshape(-1, RNN_NUM_STEPS, self.stateCnt))

    def predictOne(self, s):
        #return self.predict(s.reshape(1, self.stateCnt)).flatten()
        #return self.predict(s.reshape(1, 1, self.stateCnt)).flatten()

        # Since s is a state vector for a single step, it needs to be converted to a 1-row
        # tensor with RNN_NUM_STEPS steps. The input vector is assigned to the first time
        # step and the rest of the time steps are left with zero values.
        ts = np.zeros((len(s) * RNN_NUM_STEPS,))
        for i in range(len(s)):
            ts[i] = s[i]
        ts = ts.reshape(1, RNN_NUM_STEPS, self.stateCnt)

        #print("predictOne ts.shape: ", ts.shape)

        sequence_prediction = self.predict(ts)
        first_step_prediction = sequence_prediction[0][0]

        return first_step_prediction.flatten()


# -------------------- MEMORY --------------------------
class Memory:  # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

    def isFull(self):
        return len(self.samples) >= self.capacity


# -------------------- AGENT ---------------------------
MEMORY_CAPACITY = 1000000
#MEMORY_CAPACITY = 100000
#MEMORY_CAPACITY = 10000
#MEMORY_CAPACITY = 400
#BATCH_SIZE = 64
#BATCH_SIZE = 128
#BATCH_SIZE = 256
BATCH_SIZE = 512


#RNN_NUM_STEPS= 1                # number time steps in an RNN sequence
RNN_NUM_STEPS= 8
RNN_BATCH_SIZE = BATCH_SIZE//RNN_NUM_STEPS

ALPHA = 1.0
#ALPHA = 0.8
GAMMA = 0.99
#GAMMA = 0.95

LEARNING_RATE = 0.01

MAX_EPSILON = 1
#MIN_EPSILON = 0.1
MIN_EPSILON = 0.01  # value used in cart pole
LAMBDA = 0.001  # speed of decay

SARSA = False

STEP_LEARNING_MODE = True

NUM_STATES = 3


class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(stateCnt, actionCnt)
        self.memory = Memory(MEMORY_CAPACITY)

    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt - 1)
        else:
            return numpy.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def replay(self, batch_size):
        batch = self.memory.sample(batch_size)
        batchLen = len(batch)

        no_state = numpy.zeros(self.stateCnt)

        states = numpy.array([o[0] for o in batch])
        states_ = numpy.array([(no_state if o[3] is None else o[3]) for o in batch])

        p = agent.brain.predict(states)
        p_ = agent.brain.predict(states_)

        p = numpy.reshape(p, (-1, p.shape[2]))
        p_ = numpy.reshape(p_, (-1, p_.shape[2]))


        x = numpy.zeros((batchLen, self.stateCnt))
        y = numpy.zeros((batchLen, self.actionCnt))

        for i in range(batchLen):
            o = batch[i]
            s = o[0];
            a = o[1];
            r = o[2];
            s_ = o[3]


            # t is a vector of length actionCnt. Only one of the values in t changes:
            # the one indexed by the action value, a.
            t = p[i]   # t is a vector of length actionCnt
            if s_ is None:
                 # t[a] = r
                 t[a] = p[i][a] + ALPHA * (r - p[i][a])
            else:
                # Use Bellman equation with learning rate
                # The Q value that is going to be updated is p[i][a] - p[i] is a vector of length actionCnt.
                # Only the element indexed by the action value changes. This means the target Q value for the other
                # actions remains the same as it was. Only the target for the action a changes, for each of the rows in
                # the batch sample.

                if SARSA:
                    if random.random() < self.epsilon:
                        ind =  random.randint(0, self.actionCnt - 1)
                        max_q_val = p_[i][ind]
                    else:
                        max_q_val = numpy.amax(p_[i])
                else:
                    max_q_val = numpy.amax(p_[i])
                #t[a] = p[i][a] + ALPHA * (r + GAMMA * numpy.amax(p_[i]) - p[i][a])
                t[a] = p[i][a] + ALPHA * (r + GAMMA * max_q_val - p[i][a])

            x[i] = s
            y[i] = t

        self.brain.train(x, y)



class RandomAgent:
    memory = Memory(MEMORY_CAPACITY)

    def __init__(self, actionCnt):
        self.actionCnt = actionCnt

    def act(self, s):
        return random.randint(0, self.actionCnt - 1)

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)

    def replay(self, bath_size):
        pass


# -------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self, problem):
        self.problem = problem

        if problem == 'insurance-claim-1':
            # env is the problem environment object
            self.env = InsuranceClaim()
            self.stateCnt = self.env.numStateDimensions
            self.actionCnt = self.env.numActions

        else:
            self.env = gym.make(problem)
            self.stateCnt = self.env.observation_space.shape[0]
            self.actionCnt = self.env.action_space.n

        # high = self.env.observation_space.high
        # low = self.env.observation_space.low
        #
        # self.mean = (high + low) / 2
        # self.spread = abs(high - low) / 2

    def normalize(self, s):
        return (s - self.mean) / self.spread

    def run(self, agent, learning_flag):
        s = self.env.reset()
        if NUM_STATES < self.stateCnt:
            s = s[0:NUM_STATES]  # use the first NUM_STATES in the state vector

        #s = numpy.copy(self.env.reset()) # not necessary because env.reset returns a new array
        #s = self.normalize(s)
        R = 0
        step_cnt = 0

        while True:
            #if learning_flag:
                #self.env.render()

            a = agent.act(s)
            #print("DEBUG  action: ", a)

            s_, r, done, info = self.env.step(a)
            #s_ = self.normalize(s_)


            if NUM_STATES < self.stateCnt:
                s_ = s_[0:NUM_STATES]  # use the first NUM_STATES in the state vector


            if done:  # terminal state
                s_ = None


            agent.observe((s, a, r, s_))
            #agent.observe((s, a, r_adjusted, s_))

            # learn at each step
            if STEP_LEARNING_MODE:
                if learning_flag:
                    agent.replay(BATCH_SIZE)

            s = s_
            #s = numpy.copy(s_) # not necessary because env.step returns a new array
            R += r

            step_cnt = step_cnt + 1
            if done:
                #print("******************************************************Total reward:", R)

                if not STEP_LEARNING_MODE:
                    if learning_flag:
                        agent.replay(BATCH_SIZE)
                break
        return R



# --------------------MovingAverage-----------------------------
class MovingAverage:
    window = []

    def __init__(self, period):
        self.period = period
        self.sum = 0
        self.window = []

    def new_num(self, num):
        self.sum += num;
        self.window.append(num);
        if len(self.window) > self.period:
            self.sum -= self.window.pop(0)

    def get_avg(self):
        if len(self.window) == 0:
            return 0;  # technically the average is undefined
        else:
            return self.sum / len(self.window)





# -------------------- MAIN ----------------------------
#PROBLEM = 'MountainCar-v0'
PROBLEM = 'CartPole-v0'
env = Environment(PROBLEM)

random.seed(5)


#agent = Agent(env.stateCnt, env.actionCnt)
agent = Agent(NUM_STATES, env.actionCnt)
randomAgent = RandomAgent(env.actionCnt)

episodes_count = 0

tot_rwd_moving_average = MovingAverage(100)

optimum_flag = 0


try:
    #Data gathering with random policy until memory is full
    #i = 0
    #while randomAgent.memory.isFull() == False:
    #    env.run(randomAgent, False)
    #     i = i + 1
    #print("Completed random episodes, number: ", i)

    for i in range(BATCH_SIZE):
        env.run(randomAgent, False)

    agent.memory = randomAgent.memory
    randomAgent = None

    while True:
        episode_reward = env.run(agent, True)
        tot_rwd_moving_average.new_num(episode_reward)
        print(
            "episode: ", episodes_count, " episode reward: ", episode_reward, " moving rwd average: ",
            tot_rwd_moving_average.get_avg())

        if PROBLEM == 'insurance-claim-1':
            if (tot_rwd_moving_average.get_avg() >= -20) and (optimum_flag == 0):
                optimum_flag = 1
                print (
                    "********************************************************* Successful moving average of -20 or higher:",
                    tot_rwd_moving_average.get_avg())
                # exit(0)

                # modifiy the conditional state dimension in the business process simulator
                env.env.activate_conditional_dimension()

        elif PROBLEM == 'MountainCar-v0':
            if (tot_rwd_moving_average.get_avg() >= -110) and (optimum_flag == 0):
                # The MountainCar problem is solved when a moving average of total episode reward over
                # the past 100 episodes is -110 or larger.
                optimum_flag = 1
                print (
                    "********************************************************* Successful moving average of -110 or higher:",
                    tot_rwd_moving_average.get_avg())
                exit(0)

        elif PROBLEM == 'CartPole-v0':
            if (tot_rwd_moving_average.get_avg() >= 195) and (optimum_flag == 0):
                # The CartPole problem is solved when a moving average of total episode reward over
                # the past 100 episodes is 195 or larger.
                optimum_flag = 1
                print (
                    "********************************************************* Successful moving average of 195 or higher:",
                    tot_rwd_moving_average.get_avg())
                exit(0)
        episodes_count += 1

finally:
    agent.brain.model.save("MountainCar-basic.h5")

