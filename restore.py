import os
from pathlib import Path

def restore_from_summary(summary_file, output_dir='.'):
    with open(summary_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 定位“文件内容列表：”所在行
    content_start_idx = None
    for i, line in enumerate(lines):
        if '文件内容列表：' in line:
            content_start_idx = i + 1
            break

    if content_start_idx is None:
        print("未找到 '文件内容列表：'，请检查文件格式。")
        return

    i = content_start_idx
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('文件: '):
            file_path = line[4:].strip()
            i += 1
            # 跳过开头可能的分隔线（40个短横线）
            while i < len(lines) and lines[i].strip() == '-'*40:
                i += 1

            content_lines = []
            while i < len(lines):
                current_line = lines[i]
                # 遇到下一个文件开始，停止
                if current_line.strip().startswith('文件: '):
                    break
                # 遇到结束分隔线，跳过此行并停止收集
                if current_line.strip() == '-'*40:
                    i += 1
                    break
                content_lines.append(current_line)
                i += 1

            # 组合内容，去除末尾多余空行
            content = ''.join(content_lines).rstrip('\n')
            full_path = Path(output_dir) / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as out_f:
                out_f.write(content)
            print(f"已恢复: {full_path}")
        else:
            i += 1

    print(f"恢复完成，共处理 {len(list(Path(output_dir).rglob('*.py')))} 个文件。")

if __name__ == '__main__':
    restore_from_summary('log.txt')