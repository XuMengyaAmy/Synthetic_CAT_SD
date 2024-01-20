import synapseclient 
import synapseutils 
 
syn = synapseclient.Synapse() 
syn.login('SuAnYa','YA10260923') 
files = synapseutils.syncFromSynapse(syn, 'syn18779624') 