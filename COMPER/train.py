import sys
from random import randrange
import click
import getpass
from comper.agents.train_agent import Agent
from comper.config.optimize import TensorFlowSettings


@click.command()
@click.option("--rom",type=str, default="freeway",show_default=True)
@click.option("--logdir", type=str, default='dev',show_default=True)
@click.option("--logfreq", type=int, default=100,show_default=True)
@click.option("--net_param_dir", type=str, default='dev',show_default=True)
@click.option("--upd_target_freq", type=int, default=100,show_default=True)
@click.option("--upd_q_freq", type=int, default=4,show_default=True)
@click.option("--save_param_frq", type=int, default=1000,show_default=True)
@click.option("--lr_start_it", type=int, default=100,show_default=True)
@click.option("--memorydir", type=str, default='dev',show_default=True)
@click.option("--maxtotalframes", type=int, default=100000,show_default=True)
@click.option("--frames_ep_decay", type=int, default=90000,show_default=True)
@click.option("--persist_memories", is_flag=True,show_default=True)
@click.option("--no_save_params", is_flag=True,show_default=True)
@click.option('--framesmode', type=click.Choice(['single', 'staked'], case_sensitive=False),default='staked',show_default=True)

def main(rom,logdir, logfreq, net_param_dir, upd_target_freq, upd_q_freq,save_param_frq, lr_start_it, memorydir, 
        maxtotalframes,frames_ep_decay,persist_memories,no_save_params,framesmode):
     
    rom_file = str.encode("/home/"+getpass.getuser()+"/COMPER-RELEASE-CODE/COMPER/rom/"+rom+".bin")    
    agent = Agent(rom_name = rom,                  
                  rom_file_path=rom_file,
                  maxtotalframes=maxtotalframes,
                  frames_ep_decay = frames_ep_decay,
                  train_frequency=upd_q_freq,
                  update_target_frequency=upd_target_freq,                  
                  learning_start_iter=lr_start_it,
                  log_frequency=logfreq,                  
                  log_dir=logdir,
                  nets_param_dir=net_param_dir,
                  memory_dir=memorydir,
                  save_states_frq=save_param_frq,
                  persist_memories=persist_memories,
                  save_networks_weigths=not no_save_params,                  
                  framesmode = framesmode)
    agent.train_agent()
   

if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(type(e),e)           
            
