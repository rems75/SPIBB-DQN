import numpy as np


class helicopter_state():
	def __init__(self, x = -1, y = -1, vx = -1, vy= -1):
		self.x = x
		self.y = y
		self.vx = vx
		self.vy = vy

	def tovec(self):
		return np.array([self.x, self.y, self.vx, self.vy])

	def print(self):
		print('state = (' + str(self.x) + ', ' + str(self.y) + ', ' + str(self.vx) + ', ' + str(self.vy) + ')')


class helicopter_action():
	def __init__(self, ax=0, ay=0, rand=False, fromint=False):
		if rand:
			self.ax = np.random.randint(3) - 1
			self.ay = np.random.randint(3) - 1
		elif fromint:
			self.ax = ax // 3 - 1 # division d'entier
			self.ay = ax % 3 - 1 # reste de la division
		else:
			self.ax = ax
			self.ay = ay

	def helicopter_action_to_int(self):
		return (self.ax + 1)*3 + self.ay + 1

	def print(self):
		print('action = (' + str(self.ax) + ', ' + str(self.ay) + ')')


class helicopter_env():
	def __init__(self, time_step=0.1, size=10, noise=0.025, noise_v=0.05, v_max=1, a_max=1, log=False,
							 episode_max_len=100, seed=123, noise_factor=1):

		np.random.seed(seed=seed)

		self.time_step = time_step
		self.size = size # definit la taille de la zone de depart et d'arrivee (1/size)
		self.noise = noise * noise_factor
		self.noise_v = noise_v * noise_factor
		self.v_max = v_max
		self.a_max = a_max
		self.log = log
		self.state_shape = [4]
		self.nb_actions = 9
		self.frame_skip = 1
		self.episode_max_len = episode_max_len

		print("Using a noise factor of {}. Final noise: {} noise_v: {}".format(noise_factor, self.noise, self.noise_v))

	def get_lives(self): # for compatibility
		return 0

	def reset(self):
		# initialize the starting state: [x(0), y(0), v_x(0), v_y(0)]
		s_0 = helicopter_state(np.random.rand()/self.size, np.random.rand()/self.size, 2*np.random.rand()-1, 2*np.random.rand()-1)
		# sys.stdout.write('state = (' + str(s_0.x) + ', ' + str(s_0.y) + ', ' + str(s_0.vx) + ', ' + str(s_0.vy) + ') ')
		# sys.stdout.write('state = ({:.2f}, {:.2f}, {:.2f}, {:.2f}) '.format(
		# 	s_0.x, s_0.y, s_0.vx, s_0.vy))
		# if self.log:
		# s_0.print()
		self.current_state = s_0
		return self.get_state()

	def step(self, action):
		a = helicopter_action(ax=action, fromint=True)
		self.current_state, r, term = self.transition(self.current_state, a)
		# if term:
		# 	print(' ; ' + str(r), end='', flush=True)
		return self.current_state.tovec(), r, term, 0 # (the last argument is for compatibility)

	def get_state(self):
		return self.current_state.tovec()

	def transition(self, s, a):
		sp = helicopter_state()
		sp.x = s.x + s.vx*self.time_step*self.v_max + 0.5*a.ax*np.abs(a.ax)*(self.a_max*self.time_step)**2 + np.random.normal(0, self.noise)
		sp.y = s.y + s.vy*self.time_step*self.v_max + 0.5*a.ay*np.abs(a.ay)*(self.a_max*self.time_step)**2 + np.random.normal(0, self.noise)
		sp.vx = s.vx + a.ax*self.a_max*self.time_step + np.random.normal(0, self.noise_v)
		sp.vy = s.vy + a.ay*self.a_max*self.time_step + np.random.normal(0, self.noise_v)
		r = 0
		term = False
		if sp.x>=1 or sp.x<=0 or sp.y>=1 or sp.y<=0:
			r = min(10, max(-1, np.sqrt(1/((sp.x - 1)**2 + (sp.y - 1)**2)) - 4))
			term = True # out of bounds
		# if sp.x<1 and sp.x>1-1/self.size and sp.y<1 and sp.y>1-1/self.size:
		# 	r = 1
		# 	term = True # goal reached
		if sp.vx>=1 or sp.vx<=-1 or sp.vy>=1 or sp.vy<=-1:
			r = -1
			term = True # engine explodes (overrides other exceptions)
		if self.log:
			self.view_transition(s,a,r,sp,term)
		return sp, r, term # returns next state, immediate reward, and termination

	def view_transition(self,s,a,r,sp,term):
		s.print()
		a.print()
		print("reward = " + str(r))
		sp.print()
		print("termination = " + str(term))
		print()

	def view_transitions(self,ss,actions,rs,sps,terms):
		for i in range(len(ss)):
			self.view_transition(ss[i],actions[i],rs[i],sps[i],terms[i])