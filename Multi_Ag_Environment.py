#!/usr/bin/env python
# coding: utf-8

# In[6]:


#Import Packages
import functools
import random
from copy import copy
import pettingzoo
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete
from pettingzoo import ParallelEnv
import matplotlib.pyplot as plt
from pettingzoo.test import parallel_api_test
from gymnasium import spaces


# In[9]:


#Environment Definition
#Need parallel environment for this as multiple agents can act at once
from pettingzoo import ParallelEnv
class CustomEnvironment(ParallelEnv):
    metadata = {
        "name": "multi_satellite_downlink_v0",
    }
    def __init__(self):
        """The init method takes in environment arguments.
        
        
        These attributes should not be changed after initialization.
        """
        self.master_contact_plan = None
        self.satellite_plan = None
        self.satellite_data=[None,None,None]#Data that we must transmit to the ground
        self.timestep = None#Tracks where in the contact plan we are
        self.possible_agents = ["satellite1","satellite2","satellite3"]#Defines the names of the agents which we will use throughout the code to identify
        self.delivery_ratio=None#Metric that defines the overall delivery ratio
        self.energy_efficiency=None#Metric that defines the overall energy efficiency
        self.energy_expenditure=None #Metric that tracks the overall energy expenditure
        self.delivered_packets=None #Metric that tracks the overall number of packets delivered
        self.initial_satellite_data=None#Stores the initial data volume values to allow for delivery ratio calcs later
        self.original_matrix=None
        self.total_initial_data_volume=None#Stores sum of initial satellite data volumes
        self.initial_data_volume=self.initial_data_volume = {
    agent: [None] for agent in self.possible_agents
}
        self.sat_energy_expended=self.sat_energy_expended = {
    agent: [None] for agent in self.possible_agents
}
        self.sat_packets_delivered=self.sat_packets_delivered = {
    agent: [None] for agent in self.possible_agents
}
        self.obs_satellite_data=self.obs_satellite_data = {
    agent: [None, None, None] for agent in self.possible_agents
}
        #obs_space=spaces.Dict({
        #    "central_observation_matrix": spaces.Box(low=0.0, high=1.0, shape=(30, 5), dtype=np.float32),
        #    "remaining_satellite_data": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        #    "current_timestep": spaces.Box(low=0.0, high=32, shape=(1,), dtype=np.float32),
        #    "action_mask": spaces.MultiBinary(2),
        #})
        self.observation_spaces = {agent: self.observation_space for agent in self.possible_agents}
        self.action_spaces = {agent: self.action_space for agent in self.possible_agents}

    def reset(self, seed=None, options=None):
        """Reset set the environment to a starting point.
       
        It needs to initialize the following attributes:
        - agents
        - timestamp
        - Initial Data Volume
        - Master Contact Plan
        - Number of Delivered Packets
        - Energy Expenditure
        - Delivery Ratio
        - Energy Efficiency
        - observation
        - infos

        And must set up the environment so that render(), step(), and observe() can be called without issues.
        """
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.agents = copy(self.possible_agents)
        self.timestep = 0
        #Define the initial data volume at each satellite
        for i in range(3):
            self.satellite_data[i]=random.randint(5,100)
        print(self.satellite_data)
        self.initial_satellite_data=0
        for i in range(len(self.satellite_data)):
            self.initial_satellite_data=self.initial_satellite_data+self.satellite_data[i]
        #Define the master contact plan
        #Contains 30 contacts (total), weather values for each contact and which contacts are shared
            
        #Code below was written with a template created with ChatGPT
        nodes = ['Satellite1', 'Satellite2', 'Satellite3']
        max_rows_per_node = 10
        node_usage = {node: 0 for node in nodes}
        
        matrix = []
        
        row_index = 1
        while any(count < max_rows_per_node for count in node_usage.values()):
            # Get nodes that still have availability
            available_nodes = [node for node in nodes if node_usage[node] < max_rows_per_node]
        
            # Determine how many nodes to assign in this row
            max_possible = len(available_nodes)
            num_nodes_in_row = random.randint(1, max_possible)
        
            # Select nodes for the row
            selected_nodes = random.sample(available_nodes, num_nodes_in_row)
        
            # Update usage count
            for node in selected_nodes:
                node_usage[node] += 1
        
            # Create and store the row
            row = [random.random(), 10, selected_nodes]
            matrix.append(row)
        
            row_index += 1
        # Output the matrix
        #for row in matrix:
            #print(row)
        
        # Show final node usage
        #print("\nFinal node usage:")
        #for node, count in node_usage.items():
            #print(f"{node}: {count}")
        
        #print(f"\nTotal rows generated: {len(matrix)}")

        #Next we need either divide up the master contact plan during observations or 
        #Pad the complete matrix up to 30
        
        if len(matrix)<30:
            rows, cols = 30, 3
            new_matrix = [[None for _ in range(cols)] for _ in range(rows)]
            for i in range(30):
                if i>(len(matrix)-1):
                    new_matrix[i]=[-1,-1,-1]
                else:
                    #print(i)
                    #print(new_matrix)
                    new_matrix[i]=matrix[i]
        
        
        else:
            new_matrix=matrix
            #print(new_matrix)
        print("New Matrix")
        print(new_matrix)
        self.original_matrix=new_matrix
        #Next we must convert the new matrix into a format that can be loaded into the observation space effectively
        
        print("Contact Plan Formatting")
        rows, cols = 30, 5
        obs_matrix = [[None for _ in range(cols)] for _ in range(rows)]
        for i in range(30):
            for j in range(3):
                if j<2:
                    obs_matrix[i][j]=new_matrix[i][j]
                elif j==2:
                    current_element=new_matrix[i][j]
                    print("Current Element initial")
                    print(current_element)
                    encoded_row=[0,0,0]
                    if current_element!=-1:
                        for k in range(len(current_element)-1):
                            print("current element")
                            print(current_element[k])
                            if current_element[k]=='Satellite1':
                                encoded_row[0]=1
                                
                            elif current_element[k]=='Satellite2':
                                encoded_row[1]=1
                            elif current_element[k]=='Satellite3':
                                encoded_row[2]=1
                        print("Modified encoded row")
                        print(encoded_row)
                    else:
                        encoded_row=[0,0,0]
                    for k in range(2,5):
                        print("Encoded Row")
                        print(encoded_row)
                        obs_matrix[i][k]=encoded_row[k-2]
        print("Obs matrix")
        print(obs_matrix)
        #Next we must define the action mask 
        action_mask=[[0,0],[0,0],[0,0]]
        if obs_matrix[1][2]==1:
            action_mask[0]:[1,1]
        if obs_matrix[1][3]==1:
            action_mask[1]:[1,1]
        if obs_matrix[1][4]==1:
            action_mask[2]:[1,1]
        #Next need to configure action mask as dictionary
        obs_action_mask = {a: action_mask[i] for i, a in enumerate(self.agents)}
        #for a in self.agents:
        #    obs_action_mask=dict[a:None]
        #print(obs_action_mask)
        #i=0
        #for a in self.agents:
        #    obs_action_mask[a]=action_mask[i]
        #    i=i+1
        #print(obs_action_mask)


        
                #randomly allocate weather conditions and if the contact is shared
        self.timestep=1
        self.master_contact_plan=obs_matrix        
        #print(self.master_contact_plan)
        #Set delivery ratio for each satellite
        self.delivery_ratio=[None,None,None]
        self.delivered_packets=0
        #Set energy expenditure for each satellite
        self.energy_efficiency=[None,None,None]
        self.energy_expenditure=0
        self.sat_energy_expended=[0,0,0]
        self.initial_satellite_data_volume=[0,0,0]
        self.sat_packets_delivered=[0,0,0]
        for a in self.agents:
            self.obs_satellite_data[a]=self.satellite_obs_update(a)
            print("Initial Satellite Observations")
            print(self.obs_satellite_data[a])
        
        #Implement observations
        #Each satellite has access to the following observations
        #The total contact matrix
        #The amount of data remaining in the each satellite (at the last point we contacted the ground)
        #The current timestep
        #The action mask
        

        
        observations = {
            a: {
                "central_observation_matrix":obs_matrix,
                "remaining_satellite_data":self.obs_satellite_data[a],
                "current_timestep":self.timestep,
                "action_mask":obs_action_mask[a]
            }
            for a in self.agents
        }

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}
        print("self.agents:", self.agents)
        print("Returning observations for:", list(observations.keys()))
        print("Expected agents:", self.agents)
        missing = [agent for agent in self.agents if agent not in observations]
        if missing:
            print("❌ Missing observations for agents:", missing)
        return observations, infos

    def step(self, actions):
        """Takes in an action for the current agent (specified by agent_selection).

        Needs to update:
        - prisoner x and y coordinates
        - guard x and y coordinates
        - terminations
        - truncations
        - rewards
        - timestamp
        - infos

        And any internal state used by observe() or render()
        """
        # Execute actions
        #First we must get the actions for each satellite
        satellite_actions = [actions[a] for a in ["satellite1", "satellite2", "satellite3"]]
        #print("Satellite Actions")
        #print(satellite_actions)
        #Next we update the delivery and energy expenditure for all satellites
        
        
        #This logic was partially aided by ChatGPT (used to it to get basic structure of checking if satellites are competing for ground stations, however reward and majority of logic is my own)
        delivered_packets=0
        excess_energy=0
        total_ones = satellite_actions.count(1)
        #Next we check if there are any collisions between the agents
        collision_matrix=self.checkActionConflict(self.timestep,satellite_actions)


        rewards={}
        i=0
        #Need to fix this
        for a in self.agents:
            current_value = actions[a]
            other_ones_exist = (total_ones - (1 if current_value == 1 else 0)) > 0
            
            if collision_matrix[i]==1:
                #We have a conflict, no packets are sent and the satellite receives a large penalty
                rewards[a]=-10
                print("Collision between satellites")
                print(rewards)
                self.updateEnergyMetrics()
            else:
                if current_value==0:
                    rewards[a]=0
                    print("Satellite does not send data")
                    print(rewards)
                    
                else:
                    print("Master Contact Plan")
                    print(self.master_contact_plan)
                    print("Weather Conditions")
                    print(self.master_contact_plan[self.timestep][0])
                    print("Remaining Packets to send for current agent")
                    print(self.satellite_data[i])
                          
                    delivered_packets,excess_energy=self.updateDeliveryandEnergy(self.master_contact_plan[self.timestep][0],10,self.satellite_data[i])
                    if delivered_packets >0:
                        #First we need to update the buffers of the satellite
                        self.satBufferUpdate(i,delivered_packets)
                        #Update the Observation Space for the current agent
                        self.updateEnergyMetrics(i,excess_energy)
                        self.updateDeliveryMetrics(i,delivered_packets)
                        self.obs_satellite_data[a]=self.satellite_obs_update(a)
                        #Positive Reward
                        rewards[a]=delivered_packets/(excess_energy+1)
                        print("Satellite successfully delivers data")
                        print(rewards)
                        #Next need to update the observed satellite data values for each satellite
                        interim_satellite_data_observations=self.satellite_obs_update(a)
                    else:
                        rewards[a]=-1
                        print(rewards)
                        print("Satellite fails to deliver data")
            i=i+1
       
        
        terminations={a:False for a in self.agents}
        truncations={a: False for a in self.agents}
        #Next we must update the parameters changed in step function
        self.updateTimestep()
        #Next we must update the number of delivered packets and energy expenditur
        #Next we update the contact plan representation
        #Outline conditions for terminations
        if self.timestep>=30 :
        #All satellites have 
            truncations={a: True for a in self.agents}
            new_matrix=self.original_matrix
            rows, cols = 30, 5
            obs_matrix = [[None for _ in range(cols)] for _ in range(rows)]
            for i in range(30):
                for j in range(3):
                    if j<2:
                        obs_matrix[i][j]=new_matrix[i][j]
                    elif j==2:
                        current_element=new_matrix[i][j]
                        #print(current_element)
                        encoded_row=[0,0,0]
                        if current_element!=-1:
                            for k in range(len(current_element)-1):
                                if current_element[k]=="satellite1":
                                    encoded_row[0]=1
                                elif current_element[k]=="satellite2":
                                    encoded_row[1]=1
                                elif current_element[k]=="satellite3":
                                    encoded_row[2]=1
                        else:
                            encoded_row=[0,0,0]
                        for k in range(2,5):
                            #print("Encoded Row")
                            #print(encoded_row)
                            obs_matrix[i][k]=encoded_row[k-2]
            
            #print("Obs matrix")
            #print(obs_matrix)
            #Next we must define the action mask 
            action_mask=[[0,0],[0,0],[0,0]]
            if obs_matrix[29][2]==1:
                action_mask[0]:[0,0]
            if obs_matrix[29][3]==1:
                action_mask[1]:[0,0]
            if obs_matrix[29][4]==1:
                action_mask[2]:[0,0]
            obs_action_mask = {a: action_mask[i] for i, a in enumerate(self.agents)}
        elif self.original_matrix[self.timestep][0]==-1:
            truncations={a: True for a in self.agents}
            new_matrix=self.original_matrix
            rows, cols = 30, 5
            obs_matrix = [[None for _ in range(cols)] for _ in range(rows)]
            for i in range(30):
                for j in range(3):
                    if j<2:
                        obs_matrix[i][j]=new_matrix[i][j]
                    elif j==2:
                        current_element=new_matrix[i][j]
                        #print(current_element)
                        encoded_row=[0,0,0]
                        if current_element!=-1:
                            for k in range(len(current_element)-1):
                                if current_element[k]=="satellite1":
                                    encoded_row[0]=1
                                elif current_element[k]=="satellite2":
                                    encoded_row[1]=1
                                elif current_element[k]=="satellite3":
                                    encoded_row[2]=1
                        else:
                            encoded_row=[0,0,0]
                        for k in range(2,5):
                            #print("Encoded Row")
                            #print(encoded_row)
                            obs_matrix[i][k]=encoded_row[k-2]
            
            #print("Obs matrix")
            #print(obs_matrix)
            #Next we must define the action mask 
            action_mask=[[0,0],[0,0],[0,0]]
            if obs_matrix[29][2]==1:
                action_mask[0]:[0,0]
            if obs_matrix[29][3]==1:
                action_mask[1]:[0,0]
            if obs_matrix[29][4]==1:
                action_mask[2]:[0,0]
            obs_action_mask = {a: action_mask[i] for i, a in enumerate(self.agents)}
            
        else:
            new_matrix=self.original_matrix
            rows, cols = 30, 5
            obs_matrix = [[None for _ in range(cols)] for _ in range(rows)]
            for i in range(30):
                for j in range(3):
                    if j<2:
                        obs_matrix[i][j]=new_matrix[i][j]
                    elif j==2:
                        current_element=new_matrix[i][j]
                        #print(current_element)
                        encoded_row=[0,0,0]
                        if current_element!=-1:
                            for k in range(len(current_element)-1):
                                if current_element[k]=="satellite1":
                                    encoded_row[0]=1
                                elif current_element[k]=="satellite2":
                                    encoded_row[1]=1
                                elif current_element[k]=="satellite3":
                                    encoded_row[2]=1
                        else:
                            encoded_row=[0,0,0]
                        for k in range(2,5):
                            #print("Encoded Row")
                            #print(encoded_row)
                            obs_matrix[i][k]=encoded_row[k-2]
            
            
            #Next we must define the action mask 
            action_mask=[[0,0],[0,0],[0,0]]
            if obs_matrix[self.timestep][2]==1:
                action_mask[0]:[1,1]
            if obs_matrix[self.timestep][3]==1:
                action_mask[1]:[1,1]
            if obs_matrix[self.timestep][4]==1:
                action_mask[2]:[1,1]
            obs_action_mask = {a: action_mask[i] for i, a in enumerate(self.agents)}
        if delivered_packets==self.initial_satellite_data:
            terminations={a: True for a in self.agents}
        

        infos = {a: {} for a in self.agents}
        observations = {
            a: {
                "central_observation_matrix":obs_matrix,
                "remaining_satellite_data":self.obs_satellite_data[a],
                "current_timestep":self.timestep,
                "action_mask":obs_action_mask[a]
            }
            for a in self.agents
        }
        if any(terminations.values()) or all(truncations.values()):
            self.agents = []

        
       
        print("self.agents:", self.agents)
        print("Returning observations for:", list(observations.keys()))
        print("Expected agents:", self.agents)
        missing = [agent for agent in self.agents if agent not in observations]
        if missing:
            print("❌ Missing observations for agents:", missing)
        return observations, rewards, terminations, truncations, infos
    def getRow(self,timestep):
        return self.master_contact_plan[timestep]
    #Function to simply update timestep
    #Where the timestep is the contact in the contact plan
    def updateTimestep(self):
        self.timestep=self.timestep+1
    #Get current timestep
    def getTimestep(self):
        return self.timestep
    #In this function we obtain the number of delivered packets and excess energy expenditure 
    #If we select the given contact
    def updateDeliveryandEnergy(self,weather,length,remaining_data):
        delivered_packets=0
        excess_energy_expended=0
        random_sample=self.np_random.uniform(low=0.0, high=1.0, size=(10,)).astype(np.float32)
        for i in range(length-1):
            if remaining_data>0:
                if random_sample[i]> weather:
                    delivered_packets=delivered_packets+1
                    remaining_data=remaining_data-1
                else:
                    excess_energy_expended=excess_energy_expended+1
            else:
                excess_energy_expended=excess_energy_expended+1
        
        return delivered_packets,excess_energy_expended


    #Update the energy expenditure metrics
    def updateEnergyMetrics(self,a,energy_expended):
        #Update total energy expended
        self.energy_expenditure=self.energy_expenditure+energy_expended
        #Update the specific energy expended for the satellite
        self.sat_energy_expended[a]=self.sat_energy_expended[a]+energy_expended


        
    #Update the delivered packet metrics
    def updateDeliveryMetrics(self,a,delivered_packets):
        self.delivered_packets=self.delivered_packets+delivered_packets
        #Next update the specific packets delivered 
        self.sat_packets_delivered[a]=self.sat_packets_delivered[a]+delivered_packets
        
    #Update the satellite buffer of the agent
    def satBufferUpdate(self,agent,delivered_packets):
        #We need to convert to 
        if self.satellite_data[agent]<delivered_packets:
            self.energy_expenditure=delivered_packets-self.satellite_data[agent]
            self.satellite_data[agent]=0
            
        else:
            self.satellite_data[agent]=self.satellite_data[agent]-delivered_packets
        
    #Check if any satellites have conflicts in terms of connections
    def checkActionConflict(self,timestep,actions):
        master_observation_matrix=self.master_contact_plan
        print(timestep)
        LoS_matrix=master_observation_matrix[timestep][2:4]
        #Next we must consider the actions
        conflict_matrix=[0,0,0]
        
        for j in range(2):
            if actions[j]*LoS_matrix[j]==1:
                conflict_matrix[j]=1
        penalty_matrix=[0,0,0]
        if sum(conflict_matrix)>1:
            penalty_matrix=conflict_matrix
        return penalty_matrix
    #Update the observations of the satellites if one makes contact with the ground
    def satellite_obs_update(self,agent):
        #We take the actions and check that we established contact with the ground station
        #We update for the successful satellite the satellite data values that are known by the network
        #And we update the array that holds it for subsequent environment steps
        current_data=self.satellite_data
        self.obs_satellite_data[agent]=self.satellite_data
        return self.obs_satellite_data[agent]

    def render(self):
        print("Render not implemented yet.")
        #values = [3, 7, 2, 5, 9]
        #labels = ["satellite1 Delivery Ratio","satellite2 Delivery Ratio","satellite3 Delivery Ratio","satellite1 Energy Efficiency","satellite2 Energy Efficiency","satellite3 Energy Efficiency"]
        #plt.bar(labels, values)
        #plt.xlabel('KPI results')
        #plt.title('System Performance')
        #plt.show()
        # Observation space should be defined here.
        # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
        # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
       return spaces.Dict({
            "central_observation_matrix": spaces.Box(low=0.0, high=1.0, shape=(30, 5), dtype=np.float32),
            "remaining_satellite_data": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            "current_timestep": spaces.Box(low=0.0, high=32, shape=(1,), dtype=np.float32),
            "action_mask": spaces.MultiBinary(2),
        })
    

# Action space should be defined here.
# If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(2)

#Function to retrieve a single row from the master contact plan



# In[ ]:


#Next we evaluate the agent performance
#Start with independent PPO
#Then explore more advanced strategies

