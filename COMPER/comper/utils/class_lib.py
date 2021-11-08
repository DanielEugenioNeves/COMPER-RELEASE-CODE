
import numpy as np
import click
class Epsilon(object):
    def __init__(self, schedule_timesteps, final_p, initial_p):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction = min(1.0, float(t) / self.schedule_timesteps)
        return self.initial_p + (self.final_p - self.initial_p) * fraction


class Epsilon2(object):
    def __init__(self,init_exploration_rate,final_exploration_rate,final_exploration_frame,fraction):
        self.init_exploration_rate = init_exploration_rate
        self.final_exploration_rate = final_exploration_rate
        self.final_exploration_frame = final_exploration_frame
        self.exploration_rate = init_exploration_rate
        self.fraction = fraction
        self.rate = (self.init_exploration_rate/self.final_exploration_frame)*self.fraction

    def value(self):
        if self.exploration_rate > self.final_exploration_rate:
            self.exploration_rate = self.exploration_rate - self.rate
        else:
            self.exploration_rate = self.final_exploration_rate
        return self.exploration_rate





class Adam(object):
    def __init__(self, shape, stepsize, beta1=0.9, beta2=0.999, epsilon=1e-08):
        self.stepsize, self.beta1, self.beta2, self.epsilon = stepsize, beta1, beta2, epsilon
        self.t = 0
        self.v = np.zeros(shape, dtype=np.float32)
        self.m = np.zeros(shape, dtype=np.float32)

    def step(self, g):
        self.t += 1
        a = self.stepsize * \
            np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.v = self.beta2 * self.v + (1 - self.beta2) * (g * g)
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        step = - a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step


@click.command()
@click.option("--discount_factor", type=float, default=0.99)
@click.option("--epsilon_initial", type=float, default=1.0)
@click.option("--epsilon_final", type=float, default=0.001)
@click.option("--epsilon_fraction", type=float, default=0.10091) 
@click.option("--maxtotalframes", type=float, default=1000000)
@click.option("--initialstep", type=int, default=1)
@click.option("--maxsteps", type=int, default=1000000)

#python class_lib.py --discount_factor 0.99 --epsilon_initial 1.0 --epsilon_final 0.001 --epsilon_fraction 0.20099 --maxtotalframes 100000 --initialstep 1 --maxsteps 1000000

def main(discount_factor,epsilon_initial,epsilon_final,epsilon_fraction,maxtotalframes,initialstep,maxsteps):
    epsilon = Epsilon(schedule_timesteps=int(epsilon_fraction * maxtotalframes),initial_p=epsilon_initial,final_p=epsilon_final)
    
    stack = 1
    for itr in range(initialstep,maxsteps):
            e = epsilon.value(itr*stack)
            if(e<=0.01):
                print(e,"",itr)
                break  
    

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        print(" ")