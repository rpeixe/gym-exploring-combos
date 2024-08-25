import os
import tensorflow as tf

# Read the results in folder and make graph in tesnsorboard
def process_directory(base_dir):
    results = []
    
    for rep in os.listdir(base_dir):
        rep_path = os.path.join(base_dir, rep)
        
        if os.path.isdir(rep_path):
            for gen_folder in os.listdir(rep_path):
                gen_path = os.path.join(rep_path, gen_folder)
                
                if os.path.isdir(gen_path):
                    # Read the numbers from each fitness file
                    try:
                        with open(os.path.join(gen_path, '_bestfitness.txt')) as bf, \
                             open(os.path.join(gen_path, '_lastfitness.txt')) as lf, \
                             open(os.path.join(gen_path, '_medfitness.txt')) as mf, \
                             open(os.path.join(gen_path, '_bestcombo.txt')) as bc, \
                             open(os.path.join(gen_path, '_medbestcombo.txt')) as mbc:
                            
                            best_fitness = float(bf.read().strip())
                            last_fitness = float(lf.read().strip())
                            med_fitness = float(mf.read().strip())
                            best_combo = float(bc.read().strip())
                            med_combo = float(mbc.read().strip())
                            
                            # Extract generation number from the folder name
                            gen_num = int(gen_folder.split('_gen')[-1])
                            
                            results.append((gen_num, best_fitness, last_fitness, med_fitness, best_combo, med_combo))
                            
                    except FileNotFoundError:
                        print(f"\tWarning: Missing fitness files in {gen_path}")
    
    
    # Sort results by generation number
    results.sort(key=lambda x: x[0])
    return results

def create_tensorboard_logs(results, log_dir):
    # Create a SummaryWriter for TensorBoard logs
    writer = tf.summary.create_file_writer(log_dir)
    
    with writer.as_default():
        for gen, best_fitness, last_fitness, med_fitness, best_combo, med_combo in results:
            tf.summary.scalar("BestFitness", best_fitness, step=gen)
            tf.summary.scalar("LastFitness", last_fitness, step=gen)
            tf.summary.scalar("MedFitness", med_fitness, step=gen)
            tf.summary.scalar("BestCombo", best_combo, step=gen)
            tf.summary.scalar("MedCombo", med_combo, step=gen)
    
    writer.close()

sample_base_dir = '.teste2'
sample_log_dir = '.teste2/tensorboard_logs'

results = process_directory(sample_base_dir)

if not results:
    print("Error: No data was processed. Check if the directory structure and file names are correct.")

create_tensorboard_logs(results, sample_log_dir)
