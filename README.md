# profile_chimeras
Uses k-order Markov chains models constructed from parental genome sequences to profile chimeric genome sequences

# Dependencies
The following python libraries are required:
- numpy
- biopython
- matplotlib

# Usage
From the cloned ```profile_chimeras``` directory run:

```python profile_chimeras.py -p input_data/T7_B64_B65.fasta -c input_data/RA01_all_chimeras.fasta -j chimera_assembly_log.tsv -k <order>``` 

The script will create plots for each of the chimera profiles according to the specified order then, automatically assess the global accuracy for k = {1, 2, 3, 4} and print the results.
