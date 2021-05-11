import csv
import pandas as pd

directory = '/Users/hadi/Desktop/SOA_Complexity/PSO_Data/2400 samples/analysis/points=240/learning curves/MSE_Iter.csv'




df = pd.read_csv(directory, header=None)
rows = df.apply(lambda x: x.tolist(), axis=1)

print(rows)

"""

# Open the file in 'r' mode, not 'rb'





csv_array = []
with open(directory, 'r') as csv_file:



 
    
    
    reader = csv.reader(csv_file)
    # remove headers
    #reader.next() 
    # loop over rows in the file, append them to your array. each row is already formatted as a list.
    for row in reader:
        csv_array.append(row)
        print(row)
        break

        #csv_array = zip(csv_array[0], csv_array[1], csv_array[2],csv_array[3],csv_array[4],csv_array[5],csv_array[6],csv_array[7],csv_array[8], csv_array[9]])
    #print(csv_array[0])    """