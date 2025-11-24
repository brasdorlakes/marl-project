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


#Multi-Satellite Optical Downlink Environment
#Only works for 3 satellites (hard-coded)
#Incorporates weather condition uncertainty and competition between satellites for ground stations
#
class CustomEnvironmentFlat(ParallelEnv):
    metadata = {
        "name": "multi_satellite_downlink_v0",
    }
    def __init__(self):
        #Contact plans are a concept in delay-tolerant satellite networking that contains communication opportunity information
        #They are used to plan contacts
        self.master_contact_plan = None#Contact plan that is used for observation space and environment logic
        #Individual satellite plans (currently no different from the centralized (full observability)
        self.satellite_plan = None
        #The timestep is related to the contact plan (it tells us basically what row we are in)
        self.timestep = None#Tracks where in the contact plan we are
        #Our agents are our satellites
        self.agents = ["satellite1", "satellite2", "satellite3"]
        self.possible_agents = self.agents[:]#Defines the names of the agents which we will use throughout the code to identify
        #Each satellite has a certain amount of data in it's buffer
        self.satellite_data=self.satellite_data = {
    agent: None for agent in self.possible_agents
}
        self.delivery_ratio=None#Metric that defines the overall delivery ratio
        self.energy_efficiency=None#Metric that defines the overall energy efficiency
        self.energy_expenditure=None #Metric that tracks the overall energy expenditure
        self.delivered_packets=None #Metric that tracks the overall number of packets delivered
        self.original_matrix=None
        self.Num_of_Collisions=None#Number of times satellites are blocked because another satellite is using the ground station
        self.Total_Number_of_Contacts=None#Number of contacts used 
        self.num_of_contacts=self.num_of_contacts={
    agent: None for agent in self.possible_agents
}
        
        self.total_initial_data_volume=None#Stores sum of initial satellite data volumes
        #This variable stores the initial data volume at each satellite
        self.initial_data_volume=self.initial_data_volume = {
    agent: None for agent in self.possible_agents
}
        #This variable stores the amount of excess energy we expend at each satellite failing to transmit data
        self.sat_energy_expended=self.sat_energy_expended = {
    agent: None for agent in self.possible_agents
}
        #Amount of data delivered by each satellite
        self.sat_packets_delivered=self.sat_packets_delivered = {
    agent: None for agent in self.possible_agents
}
        #If we implement partial observability this variable would potentially be different than the true buffer occupancy of each satellite
        #This is what the satellites think each satellite is carrying in terms of data
        self.obs_satellite_data=self.obs_satellite_data = {
    agent: [None, None, None] for agent in self.possible_agents
}
        #Define the observation space for each agent
        #154 element vector which consists of flattened contact plan, satellite buffer occupancy and timestep
        self.observation_spaces = {agent: self.observation_space for agent in self.possible_agents}
        #Define the action space for each agent. Each agent has a binary action space
        self.action_spaces = {agent: self.action_space for agent in self.possible_agents}
    #Reset function resets our environment at the beginning of each episode
    def reset(self, *, seed=None, options=None):
       
        self.np_random, _ = gym.utils.seeding.np_random(seed)#To prevent errors
        self.agents = copy(self.possible_agents)#Possible unnecessary step
        self.timestep = 0#Set the timestep to the first row of the contact plan
        #Define the initial data volume at each satellite according to a uniform distribution between 5 and 100
        
        for a in self.possible_agents:
            self.satellite_data[a]=random.randint(5,100)
        
        #Here we set the initial and total data volume values according to the random values we just configured
        self.total_initial_data_volume=0
        for a in self.possible_agents:
            self.initial_data_volume[a]=self.satellite_data[a]
            self.total_initial_data_volume=self.total_initial_data_volume+self.satellite_data[a]
        #Define the master contact plan
        #Contains 30 contacts (total), weather values for each contact and which contacts are shared
        
        #Code below was written with a template created with ChatGPT
        #Basically we only want 10 contacts (transmission opportuniteies per satellite) but 
        #we need a consistent observation size for our agent so we need to randomly assign
        #contacts to the satellites but we can exceed that number of any one satellite
        nodes = ['Satellite1', 'Satellite2', 'Satellite3']
        max_rows_per_node = 10
        node_usage = {node: 0 for node in nodes}
        
        matrix = []
        
        row_index = 1
        while any(count < max_rows_per_node for count in node_usage.values()):
            
            available_nodes = [node for node in nodes if node_usage[node] < max_rows_per_node]
        
            
            max_possible = len(available_nodes)
            num_nodes_in_row = random.randint(1, max_possible)
        
        
            selected_nodes = random.sample(available_nodes, num_nodes_in_row)
            for node in selected_nodes:
                node_usage[node] += 1
        
            row = [random.random(), 10, selected_nodes]
            matrix.append(row)
        
            row_index += 1
        #If we have shared contacts (we expect this but theoretically could have entirely separate contacts)
        #However most likely case is that we have some shared contacts
        #To preserve the observation space size we pad our contact plan with -1's so hopefully the agents
        #can learn that these are not valid contacts
        if len(matrix)<30:
            rows, cols = 30, 3
            new_matrix = [[None for _ in range(cols)] for _ in range(rows)]
            for i in range(30):
                if i>(len(matrix)-1):
                    new_matrix[i]=[-1,-1,-1]
                else:
                    new_matrix[i]=matrix[i]
        
        
        else:
            new_matrix=matrix
        #This is used later to reconstruct the matrix, mostly so I can reuse code (should be a function but don't have time to mess around with this)
        self.original_matrix=new_matrix
        #Next we must convert the new matrix into a format that can be loaded into the observation space effectively
        #We encode the satellite names as 1's and 0's with 1 indicating the satellite can transmit during this transmission opportunity
        #0 indicates the satellite does not have line of sight with the ground station and hence 
        rows, cols = 30, 5
        obs_matrix = [[None for _ in range(cols)] for _ in range(rows)]
        for i in range(30):
            for j in range(3):
                if j<2:
                    obs_matrix[i][j]=new_matrix[i][j]
                elif j==2:
                    current_element=new_matrix[i][j]
                    #print("Current Element initial")
                    #print(current_element)
                    encoded_row=[0,0,0]
                    if current_element!=-1:
                        for k in range(len(current_element)):
                            #print("current element")
                            #print(current_element[k])
                            if current_element[k]=='Satellite1':
                                encoded_row[0]=1
                            elif current_element[k]=='Satellite2':
                                encoded_row[1]=1
                            elif current_element[k]=='Satellite3':
                                encoded_row[2]=1
                        #print("Modified encoded row")
                        #print(encoded_row)
                    else:
                        encoded_row=[0,0,0]
                    for k in range(2,5):
                        #print("Encoded Row")
                        #print(encoded_row)
                        obs_matrix[i][k]=encoded_row[k-2]
        #Action mask currently is not used
        #Its main purpose is to indicate to satellites if they are trying to transmit when they don't have
        #line of sight. But in initial trials we see if the agents can figure this out for themselves (they
        #would get a negative reward from this
        action_mask=[[0,0],[0,0],[0,0]]
        if obs_matrix[0][2]==1:
            action_mask[0]=[1,1]
        if obs_matrix[0][3]==1:
            action_mask[1]=[1,1]
        if obs_matrix[0][4]==1:
            action_mask[2]=[1,1]
        #Next need to configure action mask as dictionary
        #print("Original Action Mask")
        #print(action_mask)
        obs_action_mask = {a: action_mask[i] for i, a in enumerate(self.agents)}
        #We initialize the main metrics we will use to evaluate ourmodels
        self.master_contact_plan=obs_matrix        
        #Set delivery ratio for each satellite
        #This is a metric we use for evaluation
        self.delivery_ratio=[None,None,None]
        #We initialize the delivered_packets
        self.delivered_packets=0
        #Set energy expenditure for each satellite
        self.energy_efficiency=[None,None,None]
        self.energy_expenditure=0
        self.sat_energy_expended = {
    agent: 0 for agent in self.possible_agents
}
        self.sat_packets_delivered = {
    agent: 0 for agent in self.possible_agents
}
        self.Num_of_Collisions=0
        self.Total_Number_of_Contacts=0
        self.num_of_contacts={agent:0 for agent in self.possible_agents}
        for a in self.agents:
            self.obs_satellite_data[a]=self.satellite_obs_update(a)
        #We get the info to output
        infos = self.get_info()
        #we setup observations initially as a dictionary because this was easier
        observations={agent:None for agent in self.possible_agents}
        for a in self.agents:
            dict_observations={
                "central_observation_matrix":obs_matrix,
                "remaining_satellite_data":self.obs_satellite_data[a],
                "current_timestep":self.timestep,
                "action_mask":obs_action_mask[a]
            }
            #We then take this dictionary and flatten it (without action mask for now)
            output_observations=self.flatten_obs(dict_observations)
            #Allocate flattened observations 
            observations[a]=output_observations
            #print(observations[a].shape)
        return observations, infos

    def step(self, actions):
        # Execute actions
        #First we must get the actions for each satellite
        satellite_actions = [actions[a] for a in ["satellite1", "satellite2", "satellite3"]]
        
        delivered_packets=0
        excess_energy=0
        #We first check how many satellites are trying to transmit
        total_ones = satellite_actions.count(1)
        #Next we check if there are any collisions between the agents
        collision_matrix=self.checkActionConflict(self.timestep,satellite_actions)


        rewards={}
        i=0
        #Need to fix this
        for a in self.agents:
            #The action of the agent
            current_value = actions[a]
            self.updateTotalContacts(current_value)
            #If there is a conflict for this agent
            if collision_matrix[i]==1:
                #We have a conflict, no packets are sent and the satellite receives a large penalty
                rewards[a]=-50
                #We update the energy metric for the agent and 
                self.updateEnergyMetrics(a,10)
                self.Num_of_Collisions=self.Num_of_Collisions+1
            else:
                #If the agent decides not to send, then it gets zero reward
                if current_value==0:
                    rewards[a]=0
                    
                else:
               
                    #Here according to defined logic we update the number of packets delivered and excess energy expended
                    #
                    delivered_packets,excess_energy=self.updateDeliveryandEnergy(self.master_contact_plan[self.timestep][0],10,self.satellite_data[a])
                    if delivered_packets >0:
                        #First we need to update the buffers of the satellite
                        self.satBufferUpdate(a,delivered_packets)
                        #Update the Observation Space for the current agent
                        self.updateEnergyMetrics(a,excess_energy)
                        self.updateDeliveryMetrics(a,delivered_packets)
                        self.obs_satellite_data[a]=self.satellite_obs_update(a)
                        #Calculate the fraction of the remaining data for this particular satellite
                        #To deliver
                        delivery_fraction=self.sat_packets_delivered[a]/self.initial_data_volume[a]
                            
                        #we allocate rewards according in this case according to a fairness 
                        #With satellites that have who have delivered less of their data
                        #Given greater priority
                        rewards[a]=10*(delivery_fraction+1)*(delivered_packets/10-0.5*(delivered_packets/10)*(excess_energy/(excess_energy+10)))
                      
                        #Next need to update the observed satellite data values for each satellite
                        #This is action is only important in the partially observable case
                        interim_satellite_data_observations=self.satellite_obs_update(a)
                    else:
                        #If we don't transmit successfully
                        #we punish the agent
                        rewards[a]=-5 * (0.5*(excess_energy/(excess_energy+10)))
            #Are stupid iterator
            #because I don't have time to convert everything to operate consistently according to agent names
            i=i+1
       
        
        terminations={a:False for a in self.agents}
        truncations={a: False for a in self.agents}

        #We increase the timestep value by 1
        self.updateTimestep()
        #We check if we have reached the end of our contact plan or not
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
            
       
            #Action mask currently is not used
            #Its main purpose is to indicate to satellites if they are trying to transmit when they don't have
            #line of sight. But in initial trials we see if the agents can figure this out for themselves (they
            #would get a negative reward from this
            action_mask=[[0,0],[0,0],[0,0]]
            if obs_matrix[29][2]==1:
                action_mask[0]:[0,0]
            if obs_matrix[29][3]==1:
                action_mask[1]:[0,0]
            if obs_matrix[29][4]==1:
                action_mask[2]:[0,0]
            obs_action_mask = {a: action_mask[i] for i, a in enumerate(self.agents)}
        #Or if we run into our padding we truncate our episode
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
        #Otherwise we setup the matrix for the next step
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
                        encoded_row=[0,0,0]
                        if current_element!=-1:
                            for k in range(len(current_element)-1):
                                if current_element[k]=="Satellite1":
                                    encoded_row[0]=1
                                elif current_element[k]=="Satellite2":
                                    encoded_row[1]=1
                                elif current_element[k]=="Satellite3":
                                    encoded_row[2]=1
                        else:
                            encoded_row=[0,0,0]
                        for k in range(2,5):
                        
                            obs_matrix[i][k]=encoded_row[k-2]
            
            
            # Don't use this currently
            action_mask=[[0,0],[0,0],[0,0]]
            if obs_matrix[self.timestep][2]==1:
                action_mask[0]=[1,1]
            if obs_matrix[self.timestep][3]==1:
                action_mask[1]=[1,1]
            if obs_matrix[self.timestep][4]==1:
                action_mask[2]=[1,1]
            obs_action_mask = {a: action_mask[i] for i, a in enumerate(self.agents)}
        #If we deliver all the packets we terminate the episode
        if self.delivered_packets==self.total_initial_data_volume:
            terminations={a: True for a in self.agents}
        
        infos = self.get_info()
        for a in self.agents:
            self.obs_satellite_data[a]=self.satellite_obs_update(a)
        observations={agent:None for agent in self.possible_agents}
        for a in self.agents:
            dict_observations={
                "central_observation_matrix":obs_matrix,
                "remaining_satellite_data":self.obs_satellite_data[a],
                "current_timestep":self.timestep,
                "action_mask":obs_action_mask[a]
            }
            output_observations=self.flatten_obs(dict_observations)
            observations[a]=output_observations
            print(observations[a].shape)
        if any(terminations.values()) or all(truncations.values()):
            self.agents = []
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
    def updateDeliveryandEnergy(self,weather,length,remaining_data):
        #Updates delivery and energy metrics according to our system model
        
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
        self.sat_energy_expended[a] += energy_expended


        
    #Update the delivered packet metrics
    def updateDeliveryMetrics(self,a,delivered_packets):
        #Updates total and individual satellite delivered packet totals
        self.delivered_packets=self.delivered_packets+delivered_packets
        #Next update the specific packets delivered 
        self.sat_packets_delivered[a]=self.sat_packets_delivered[a]+delivered_packets
        
    #Update the satellite buffer of the agent
    def satBufferUpdate(self,agent,delivered_packets):
        #Updates the satellite buffers
        if self.satellite_data[agent]<delivered_packets:
            self.energy_expenditure=delivered_packets-self.satellite_data[agent]
            self.satellite_data[agent]=0
            
        else:
            self.satellite_data[agent]=self.satellite_data[agent]-delivered_packets
        
    #Check if any satellites have conflicts in terms of connections
    def checkActionConflict(self,timestep,actions):
        #Checks for conflicts in satellite downlink requests
        master_observation_matrix=self.master_contact_plan
        LoS_matrix=master_observation_matrix[timestep][2:5]
        print(LoS_matrix)
        conflict_matrix=[0,0,0]
        for j in range(0,3):
            if actions[j]*LoS_matrix[j]==1:
                conflict_matrix[j]=1
        penalty_matrix=[0,0,0]
        if sum(conflict_matrix)>1:
            penalty_matrix=conflict_matrix
        return penalty_matrix
    def satellite_obs_update(self,agent):
        #We take the actions and check that we established contact with the ground station
        #We update for the successful satellite the satellite data values that are known by the network
        #And we update the array that holds it for subsequent environment steps
        current_data=self.satellite_data
        self.obs_satellite_data[agent]=self.satellite_data
        return self.obs_satellite_data[agent]
    def flatten_obs(self,obs):
        #Takes the observation configuration and flattens according to requirements
        flattened_obs=[]
        for i in range(0,len(obs["central_observation_matrix"])):
            contact_plan=obs["central_observation_matrix"]
            for j in contact_plan[i]:
                flattened_obs.append(j)
        #print("Before i")
        #print(obs["remaining_satellite_data"])
        all_agents=["satellite1","satellite2","satellite3"]
        satellite_buffer_obs=obs["remaining_satellite_data"]
        for i in all_agents:
            flattened_obs.append(satellite_buffer_obs[i])
        flattened_obs.append(obs["current_timestep"])
        return np.array(flattened_obs, dtype=np.float32)
                           

    def render(self):
        #Too hard to implement
        print("Render not implemented yet.")
        
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        
        return spaces.Box(low=-1.0,high=1.0,shape=(154,),dtype=np.float32)
    def get_info(self):
        #Info outputs metrics useful for graphing and understanding agent behaviour
        return {a:{"Delivered_Packets":self.delivered_packets,"Total_Energy_Expended":self.energy_expenditure,"True_Contact_Matrix":self.master_contact_plan,"satellite_delivered_packets":self.sat_packets_delivered[a],"satellite_energy_expended":self.sat_energy_expended[a],"satellite_initial_data_volume":self.initial_data_volume[a],"Number_of_Collisions":self.Num_of_Collisions,"Initial_Satellite_Data_Volume":self.initial_data_volume[a],"Satellite_Initial_Satellite_Data_Volume":self.initial_data_volume[a],"Total_Initial_Satellite_Data_Volume": self.total_initial_data_volume, "Total_Number_of_Contacts_Used": self.Total_Number_of_Contacts
        ,"satellite_number_of_contacts_used":self.num_of_contacts[a]} for a in self.agents}
    def updateTotalContacts(self,action):
        #Just keeping track of the total number of contacts our agents try to use
        self.Total_Number_of_Contacts+=action
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        #Satellite can either transmit or not, so action space is binary
        return Discrete(2)


# In[ ]:




