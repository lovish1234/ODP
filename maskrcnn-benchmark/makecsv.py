import csv
f = open("results/spade/11_18_ft_result.txt", "r")
dic = {"bbox":{}, "segm":{}}
name = ""
task = ""
for line in f.readlines():
    if len(line) < 2 or line[:2] == "AP":
        continue

    if line[:4] == "Task":
        task = line.strip().split(" ")[-1]
    elif "_" in line:
        name = line.strip()
    else:
        numbers = [num for num in line.strip().split(" & ")]
        dic[task][name] = numbers

o = open("results/spade/11_18_ft_result.csv", "w")

for t in dic.keys():
    o.write(t + "\n")
    for k, v in dic[t].items():
        s = ",".join(v)
        o.write(k + "," + s + "\n")