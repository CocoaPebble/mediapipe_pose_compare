import json

filepath = ""

with open(filepath, 'r') as f:
    data = json.load(f)

extra_joint_list = ['mid_hip', 'spine', 'mid_shoulder']
ref_joint_list = ['left_hip', 'right_hip', 'left_shoulder', 'right_shoulder']

first_key = list(data.keys())[0]

for index in range(len(data)):
    # for i in range(len(extra_joint_list)):

    data[str(index+int(first_key))]['mid_hip'] = [0, 0, 0]
    data[str(index+int(first_key))]['mid_shoulder'] = [0, 0, 0]
    data[str(index+int(first_key))]['spine'] = [0, 0, 0]

    for i in range(3):
        data[str(index+int(first_key))]['mid_hip'][i] = (data[str(index+int(first_key))]
                                                         ['left_hip'][i] + data[str(index+int(first_key))]['right_hip'][i]) * 0.5
        print(data[str(index+int(first_key))]['mid_hip'][i])
        data[str(index+int(first_key))]['mid_shoulder'][i] = (data[str(index+int(first_key))]
                                                              ['left_shoulder'][i] + data[str(index+int(first_key))]['right_shoulder'][i]) * 0.5
        print(data[str(index+int(first_key))]['mid_shoulder'][i])
        data[str(index+int(first_key))]['spine'][i] = (data[str(index+int(first_key))]
                                                       ['mid_shoulder'][i] - data[str(index+int(first_key))]['mid_hip'][i]) * 0.25
        print(data[str(index+int(first_key))]['spine'][i])


outpath = "test\\rolljump2json_updated.json"
# outpath = "groundtruth\\amassJumpRope0012json_updated.json"

with open(outpath, 'w') as f:
    json.dump(data, f, indent=4)
