import json
import sys

def filter_tutsed2016(input_file, output_file):
    """
    Filter JSON entries that contain 'TUTSED2016' in the 'audio' field.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
    """
    try:
        # Read the input JSON file
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Filter entries containing 'TUTSED2016' or 'TUTSED2017' in the audio field
        filtered_data = [
            entry for entry in data 
            if 'audio' in entry and any(
                keyword in entry['audio'] for keyword in ['TUTSED2016', 'TUTSED2017']
            )
        ]
        
        count_2016 = sum(1 for e in filtered_data if 'TUTSED2016' in e['audio'])
        count_2017 = sum(1 for e in filtered_data if 'TUTSED2017' in e['audio'])

        # Write filtered data to output file
        with open(output_file, 'w') as f:
            json.dump(filtered_data, f, indent=2)
        
        print(f"✓ TUTSED2016 entries: {count_2016}")
        print(f"✓ TUTSED2017 entries: {count_2017}")
        print(f"✓ Total filtered: {len(filtered_data)} entries")
        print(f"✓ Output saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: '{input_file}' is not a valid JSON file")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python filter_tutsed2016.py <input_file.json> <output_file.json>")
        print("Example: python filter_tutsed2016.py data.json tutsed2016_only.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    filter_tutsed2016(input_file, output_file)
