import os
import yaml
from main.optimizer import Oracle
from tdc import Oracle as Evaluator
import argparse
import copy

def get_oracle_path(args,oracles):
  oracle_names = ""
  for oracle in args.oracles:
    oracle_names += oracle+"_"
    suffix = args.method + "_" + oracle_names + "0" #since it selects 0 only
  return suffix
     

def get_yaml_path(args):
     suffix = get_oracle_path(args,args.oracles)
     yaml_path = os.path.join(args.output_dir, 'results_' + suffix + '.yaml')
     return yaml_path

def load_yaml(args):
   
    yaml_path= get_yaml_path(args)
    try:
        with open(yaml_path, 'r') as file:
            mol_buffer = yaml.safe_load(file)
            print("---Reading {}---".format(yaml_path))
            for oracle_name in args.oracles:
                copy_args = argparse.Namespace(**{k: copy.deepcopy(v) for k, v in vars(args).items()})
                copy_args.oracles = [oracle_name]
                print("---processing: {}---".format(oracle_name))
                #print("copy_args", copy_args)
                oracle = Oracle(args=copy_args,mol_buffer={}) #create copy oracle with singleton name
                oracle.assign_evaluator([Evaluator(name=oracle_name)])
                #print("initial mol_buffer: ",oracle.mol_buffer)
                #print("evaluator: ",oracle.evaluators[0].name)
                #print(list(mol_buffer.keys()))
                scores = oracle(list(mol_buffer.keys())) #populate oracle with all the keys from multiobjective
                #print("scores from oracle: ",scores)
                #print("mol_buffer obj: ",mol_buffer)
                #scores_ = [float(oracle.evaluators[0](smi)) for smi in mol_buffer.keys()]
                oracle.sort_buffer()
                #print("scores directly from evaluator: ",scores_)
                #print("current mol_buffer: ",oracle.mol_buffer)
                oracle.log_intermediate(finish=True)
                oracle.save_result(suffix=get_oracle_path(copy_args,oracle_name))
            
                
            
    except FileNotFoundError:
        print("Error: {yaml_path} not found.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
    
    
    