import sys

from random import randrange

from ale_python_interface import ALEInterface

#if len(sys.argv) <2:
	#print('Usage: %s rom_file' % sys.argv[0])
	#sys.exit()

ale = ALEInterface()

#-----Get & Set desired settings-------

ale.setInt(b'random_seed', 123)

#-----Set USE_SDL to true to display the screen. ALE must be compiled with SDL enabled for this work.

USE_SDL = True

if USE_SDL:
	ale.setBool(b'sound', True)
	ale.setBool(b'display_screen', True)

#-----Load the ROM file

rom_file = str.encode("rom/pong.bin")#str.encode(sys.argv[1])
ale.loadROM(rom_file)

#----Get the list of legal actions

legal_actions = ale.getLegalActionSet()

#----Play 10 episodes

for episode in range(10):
	total_reward=0
	while not ale.game_over():
		a = legal_actions[randrange(len(legal_actions))]
		# ------Apply an action and get the resulting reward
		reward = ale.act(a)
		total_reward +=reward
	print('Episode %d ended whith score %d' % (episode, total_reward))
	ale.reset_game()
