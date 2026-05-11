import os
import shutil


def process_kg_files(data):
    # 初始化集合用于存储唯一的实体和关系
    entities = set()
    relations = set()

    # 输入文件列表
    files = [f'{data}/train.txt', f'{data}/valid.txt', f'{data}/test.txt']

    # 处理每个文件
    for file_name in files:
        try:
            with open(file_name, 'r', encoding='utf-8') as file:
                for line in file:
                    # 跳过空行
                    if line.strip() == '':
                        continue

                    # 将行拆分为头实体、关系和尾实体
                    parts = line.strip().split('\t')
                    if len(parts) != 3:
                        print(f"警告: 跳过{file_name}中格式错误的行: {line}")
                        continue

                    head, relation, tail = parts

                    # 将实体和关系添加到集合中
                    entities.add(head)
                    entities.add(tail)
                    relations.add(relation)
        except FileNotFoundError:
            print(f"警告: 文件{file_name}未找到。")

    # 将集合转换为排序列表以获得一致的编号
    entities_list = sorted(list(entities))
    relations_list = sorted(list(relations))

    # 写入entities.txt
    with open(f'{data}/entities.txt', 'w', encoding='utf-8') as file:
        for i, entity in enumerate(entities_list):
            file.write(f"{entity}\t{i}\n")

    # 写入relations.txt
    with open(f'{data}/relations.txt', 'w', encoding='utf-8') as file:
        for i, relation in enumerate(relations_list):
            file.write(f"{relation}\t{i}\n")

    print(f"处理了{len(entities_list)}个唯一实体和{len(relations_list)}个唯一关系。")

    # 复制data文件夹为data_ind
    data_ind = f"{data}_ind"

    # 如果目标文件夹已存在，先删除
    if os.path.exists(data_ind):
        shutil.rmtree(data_ind)

    # 复制整个文件夹
    shutil.copytree(data, data_ind)

    print(f"已将{data}文件夹复制为{data_ind}。")


if __name__ == "__main__":
    data = "family"
    process_kg_files(data)