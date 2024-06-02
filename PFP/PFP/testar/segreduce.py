def distribute_data(data, num_units):
    # Calculate the number of elements per segment
    segment_size = len(data) // num_units
    scattered_data = []
    
    # Distribute segments across processing units
    for i in range(num_units):
        start_index = i * segment_size
        end_index = (i + 1) * segment_size if i < num_units - 1 else len(data)
        segment = data[start_index:end_index]
        scattered_data.append(segment)
    
    return scattered_data

def op(v1, f1, v2, f2):
    # Define the operation ⊕′
    result_value = v1 if not f2 else v1 + v2
    result_flag = f1 or f2
    return (result_value, result_flag)

def scatter(data):
    # Scatter data across processing units
    scattered_data = distribute_data(data)
    return scattered_data

def local_reduction(scattered_data):
    result = []
    for segment_data in scattered_data:
        segment_result_value = 0
        segment_result_flag = False
        for value, flag in segment_data:
            segment_result_value, segment_result_flag = op(segment_result_value, segment_result_flag, value, flag)
        result.append((segment_result_value, segment_result_flag))
    return result

def gather(local_results):
    # Gather local results from all processing units
    final_result_value = 0
    final_result_flag = False
    for value, flag in local_results:
        final_result_value, final_result_flag = op(final_result_value, final_result_flag, value, flag)
    return (final_result_value, final_result_flag)

# Example usage
input_data = [(1, True), (2, True), (3, False), (4, True), (5, False)]
scattered_data = scatter(input_data)
local_results = local_reduction(scattered_data)
final_result = gather(local_results)
print(final_result)