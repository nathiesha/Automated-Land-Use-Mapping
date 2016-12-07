import csv
import sys

f = open("original.csv", "r")
try:
    reader = csv.reader(f)
    for row in reader:
        print row
finally:
    f.close()