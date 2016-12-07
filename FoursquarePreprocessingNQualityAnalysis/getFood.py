import csv

counter = 0
with open("original.csv", "rb") as inp, open("food.csv", "wb") as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[4] == "Food":
            writer.writerow(row)
            counter+=1
    print counter