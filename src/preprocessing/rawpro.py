import re
import json
import argparse
import os
from datetime import datetime


def generate_log_regex():
    """
    Generate a regular expression for parsing log lines.
    Assuming the log format is:
    Date Time,ms | Level | [Component] | Content | ClassInfo
    For example:
    2024-12-18 11:15:49,281 | INFO  | [Component Info] | Message Content | ClassInfo
    """
    log_pattern = re.compile(
        r'^(?P<Date>\d{4}-\d{2}-\d{2})\s+'
        r'(?P<Time>\d{2}:\d{2}:\d{2},\d{3})\s*\|\s*'
        r'(?P<Level>\w+)\s*\|\s*'
        r'\[(?P<Component>[^\]]+)\]\s*\|\s*'
        r'(?P<Content>[^|]+)\s*\|\s*'
        r'(?P<ClassInfo>.+)$'
    )
    return log_pattern


def generate_logformat_regex(logformat):
    """Function to generate regular expression to split log messages"""
    headers = []
    splitters = re.split(r"(<[^<>]+>)", logformat)
    regex = ""
    for k in range(len(splitters)):
        if k % 2 == 0:
            splitter = re.sub(" +", "\\\s+", splitters[k])
            regex += splitter
        else:
            header = splitters[k].strip("<").strip(">")
            regex += "(?P<%s>.*?)" % header
            headers.append(header)
    regex = re.compile("^" + regex + "$")
    return headers, regex


def parse_log_line(line, regex):
    """
    Parse a single log line and return a dictionary if it matches the standard format, otherwise return None.
    """
    match = regex.match(line)
    if match:
        log_dict = match.groupdict()
        # Combine Date and Time into a complete timestamp
        timestamp_str = f"{log_dict['Date']} {log_dict['Time']}"
        try:
            # Parse the time, removing the milliseconds part
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
            log_dict['Timestamp'] = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            log_dict['Timestamp'] = ""
        return {
            "time": log_dict.get("Timestamp", ""),
            "levels": log_dict.get("Level", ""),
            "raw_log": log_dict.get("Content", "").strip()
        }
    else:
        return None


def process_log_file(file_path, k):
    """
    Process the log file, grouping every k standard log lines and generating a corresponding list of dictionaries.
    """
    logformat = "<Date> <Time> | <Level> | [<Component>] | <Content> | <ClassInfo>"
    regex = generate_logformat_regex(logformat=logformat)
    regex = generate_log_regex()
    parsed_logs = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parsed = parse_log_line(line.strip(), regex)
            if parsed:
                parsed_logs.append(parsed)

    grouped_logs = []
    total_logs = len(parsed_logs)
    num_groups = (total_logs + k - 1) // k  # Round up
    for i in range(num_groups):
        start_idx = i * k
        end_idx = min((i + 1) * k, total_logs)
        group = parsed_logs[start_idx:end_idx]
        group_dict = {
            "idx": i + 1,
            "line_range": f"{start_idx + 1}-{end_idx}",
            "raw_log": [log["raw_log"] for log in group],
            "summary": "",
            "time": [log["time"] for log in group],
            "levels": [log["level"] for log in group],
            "parameters": [],
            "templates": []
        }
        grouped_logs.append(group_dict)

    return grouped_logs


def save_to_json(data, input_filename):
    """
    Save the processed data to a JSON file, with the file name modified based on the input file name.
    For example, input 'log.txt' -> 'log_processed.json'
    """
    base, ext = os.path.splitext(input_filename)
    output_filename = f"{base}_processed.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Processed data saved to {output_filename}")


def main():
    parser = argparse.ArgumentParser(description="Convert a log file to the specified JSON format.")
    parser.add_argument('--filename', type=str, default='../../data/raw_files/file_test.txt', help='Input log file name')
    parser.add_argument('--k', type=int, default=100, help='Divide every k lines')
    parser.add_argument('--save_filename', type=str, default='../../data_zengge/tttest.json', help='Input log file name')
    args = parser.parse_args()

    if not os.path.isfile(args.filename):
        print(f"File {args.filename} does not exist.")
        return

    grouped_logs = process_log_file(args.filename, args.k)
    save_to_json(grouped_logs, args.save_filename)


if __name__ == "__main__":
    main()