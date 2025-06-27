'''
This file extracts the ap values for the first 100 primes, conductor and root number for elliptic curves drawn from `https://math.mit.edu/~edgarc/data/ECQ6ap1e4.txt`. 
'''

def split_first_100_commas(input_str):
    '''
    Takes a string and returns the first 100 integers separated by commas.
    '''
    # Initialize variables
    parts = []
    count = 0
    start = 0
    
    # Iterate through the string to split at commas, but limit to the first 100 commas
    for i in range(len(input_str)):
        if input_str[i] == ',':
            count += 1
            if count <= 100:
                parts.append(input_str[start:i])  # Add the substring up to the comma
                start = i + 1  # Update start to the next character after the comma
            else:
                # Once we hit the 100th comma, break and add the rest of the string
                #parts.append(input_str[start:])
                break
    
    # If there are less than 100 commas, just add the remainder of the string
    #if count < 100:
    #    parts.append(input_str[start:])
    
    return parts

import pandas as pd
import ast

# Define columns
cols = [str(p + 1) for p in range(100)]
cols.append('conductor')
cols.append('root number')

def Vtob(line, cols):
    '''
    Makes sure that every line is of the expected input type. Appends root number and conductor.
    '''
    V = tuple([x for x in line.split(':')])
    if V[0] == '':
        return None
    else:
        cond = int(V[0])
    
    if V[4] == '':
        return None
    else:
        rank = int(V[4])
    
    aps = split_first_100_commas(V[7][1:]) 
    aps = list(map(int, aps))
    aps.append(cond)
    aps.append((-1)**rank)
    
    return aps

# Initialize an empty list to hold all rows
rows = []

print("Extracting data..")

# Open file and process each line
import requests

url = "https://math.mit.edu/~edgarc/data/ECQ6ap1e4.txt"
output_file = "filtered_curves.txt"

# Write this data to the output file
with requests.get(url, stream=True) as r, open(output_file, "w") as outfile:
    r.raise_for_status()
    for line in r.iter_lines():
        if not line:
            continue
        aps = Vtob(line.decode(), cols)
        if aps:
            rows.append(aps)
        outfile.write(str(aps)+"\n") 

print("Done")
