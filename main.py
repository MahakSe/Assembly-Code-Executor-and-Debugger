import ast
import copy
import sys

data_types = {"BYTE": 8, "WORD": 16, "DWORD": 32, "SBYTE": 8, "SWORD": 16, "SDWORD": 32}
instructions = {
    "INC": {"operandsNumber": 1, "same_size": None},
    "DEC": {"operandsNumber": 1, "same_size": None},
    "NEG": {"operandsNumber": 1, "same_size": None},
    "ADD": {"operandsNumber": 2, "same_size": True},
    "SUB": {"operandsNumber": 2, "same_size": True},
    "OR": {"operandsNumber": 2, "same_size": True},
    "AND": {"operandsNumber": 2, "same_size": True},
    "XOR": {"operandsNumber": 2, "same_size": True},
    "MOV": {"operandsNumber": 2, "same_size": True},
    "XCHG": {"operandsNumber": 2, "same_size": True},
    "MOVSX": {"operandsNumber": 2, "same_size": False},
    "MOVZX": {"operandsNumber": 2, "same_size": False},
    "POP": {"operandsNumber": 0, "same_size": None},
    "PUSH": {"operandsNumber": 1, "same_size": None},
    "CMP": {"operandsNumber": 2, "same_size": True},
    "JMP": {"operandsNumber": 1, "same_size": None},
    "JZ": {"operandsNumber": 1, "same_size": None},
    "JNZ": {"operandsNumber": 1, "same_size": None},
    "JC": {"operandsNumber": 1, "same_size": None},
    "JNC": {"operandsNumber": 1, "same_size": None},
    "JO": {"operandsNumber": 1, "same_size": None},
    "JNO": {"operandsNumber": 1, "same_size": None},
    "JS": {"operandsNumber": 1, "same_size": None},
    "JNS": {"operandsNumber": 1, "same_size": None},
    "JP": {"operandsNumber": 1, "same_size": None},
    "JNP": {"operandsNumber": 1, "same_size": None},
    "JE": {"operandsNumber": 1, "same_size": None},
    "JNE": {"operandsNumber": 1, "same_size": None},
    "JCXZ": {"operandsNumber": 1, "same_size": None},
    "JECXZ": {"operandsNumber": 1, "same_size": None},
    "JA": {"operandsNumber": 1, "same_size": None},
    "JNBE": {"operandsNumber": 1, "same_size": None},
    "JAE": {"operandsNumber": 1, "same_size": None},
    "JNB": {"operandsNumber": 1, "same_size": None},
    "JB": {"operandsNumber": 1, "same_size": None},
    "JNAE": {"operandsNumber": 1, "same_size": None},
    "JBE": {"operandsNumber": 1, "same_size": None},
    "JNA": {"operandsNumber": 1, "same_size": None},
    "JG": {"operandsNumber": 1, "same_size": None},
    "JNLE": {"operandsNumber": 1, "same_size": None},
    "JGE": {"operandsNumber": 1, "same_size": None},
    "JNL": {"operandsNumber": 1, "same_size": None},
    "JL": {"operandsNumber": 1, "same_size": None},
    "JNGE": {"operandsNumber": 1, "same_size": None},
    "JLE": {"operandsNumber": 1, "same_size": None},
    "JNG": {"operandsNumber": 1, "same_size": None},
    "LOOP": {"operandsNumber": 1, "same_size": None},
}
registers = {
    "EAX": [32, "EAX"],
    "EBX": [32, "EBX"],
    "ECX": [32, "ECX"],
    "EDX": [32, "EDX"],
    "EBP": [32, "EBP"],
    "ESP": [32, "ESP"],
    "ESI": [32, "ESI"],
    "EDI": [32, "EDI"],
    "AX": [16, "EAX"],
    "BX": [16, "EBX"],
    "CX": [16, "ECX"],
    "DX": [16, "EDX"],
    "BP": [16, "EBP"],
    "SP": [16, "ESP"],
    "SI": [16, "ESI"],
    "DI": [16, "ESI"],
    "AL": [8, "EAX"],
    "BL": [8, "EBX"],
    "CL": [8, "ECX"],
    "DL": [8, "EDX"],
    "AH": [8, "EAX"],
    "BH": [8, "EBX"],
    "CH": [8, "ECX"],
    "DH": [8, "EDX"]
}

REGISTERS = {"EAX": "00000000", "EBX": "00000000", "ECX": "00000000", "EDX": "00000000", "EBP": "00000000",
             "ESP": "00000000", "ESI": "00000000", "EDI": "00000000"}
FLAGS = {"carry": 0, "overflow": 0, "sign": 0, "parity": 0, "auxiliary": 0, "zero": 0}
runtime_stack = []
top = "FFFF"


def hex_to_binary(hex_str):
    try:
        # Convert the hexadecimal string to a decimal integer
        decimal_number = int(hex_str, 16)

        # Convert the decimal integer to a binary string and remove the '0b' prefix
        binary_str = bin(decimal_number)[2:]
    except ValueError:
        # Handle the case where the string is not a valid hexadecimal number
        return "Error: Invalid hexadecimal input"

    return binary_str


def binary_to_decimal(binary_str):
    try:
        # Convert the binary string to a decimal integer
        decimal_number = int(binary_str, 2)
    except ValueError:
        # Handle the case where the string is not a valid binary number
        return "Error: Invalid binary input"

    return decimal_number


def binary_to_hex(binary_str):
    try:
        # Convert the binary string to a decimal integer
        decimal_number = int(binary_str, 2)

        hex_str = hex(decimal_number)[2:].upper()

        required_length = (len(binary_str) + 3) // 4

        # Ensure the hex string is the correct length, padded with leading zeros if necessary
        hex_str = hex_str.zfill(required_length)
    except ValueError:
        # Handle the case where the string is not a valid binary number
        return "Error: Invalid binary input"

    return hex_str


def to_binary(decimal_str, capacity):
    # Convert the decimal string to an integer
    decimal_number = int(decimal_str)

    # Check if the number is negative
    if decimal_number < 0:
        # Compute the two's complement for the negative number
        binary_str = bin((1 << capacity) + decimal_number)[2:]
    else:
        # Convert the positive number to binary
        binary_str = bin(decimal_number)[2:]

    # Ensure the binary string fits within the specified capacity
    if len(binary_str) > capacity:
        # If the binary representation exceeds the capacity, truncate it
        return True, "initializer magnitude too large for specified size"
    else:
        # Otherwise, pad it with leading zeros
        binary_str = binary_str.zfill(capacity)

    return False, binary_str


def to_ten(string, base):
    if base == "o" or base == "q" or base == "O" or base == "Q":
        base = 8
    elif base == 'h':
        base = 16
    elif base == 'b':
        base = 2
    try:
        # Convert the string from the given base to a decimal integer
        result = int(string, base)
    except ValueError:
        # Handle the case where the string is not a valid number in the given base
        return True, "Error: Invalid input for the given base"

    return False, result


def parse_values(input_str, data_type):
    # Replace commas followed by spaces with just commas
    input_str = input_str.replace(", ", ",")

    # Split by comma and strip each part
    parts = [part.strip() for part in input_str.split(",")]

    i = 0
    for part in parts:
        if part.isdigit():  # "1729"
            error, result = to_binary(part, data_type)
            if not error:
                parts[i] = result
            else:
                return result, None

        elif part[1:].isdigit():  # "-1729" or "-17281d"
            error, result = to_binary(part, data_type)
            if not error:
                parts[i] = result
            else:
                return result, None

        elif part[0:-1].isdigit():  # "10101b" or "AB10h" or "12345678o"
            base = part[-1]
            if base == "b":
                if len(part[0:-1]) > data_type:
                    return "initializer magnitude too large for specified size", None
                result = part[0:-1].zfill(data_type)
                parts[i] = result

            elif base == "o" or base == "q" or base == "O" or base == "Q":
                error2, res = to_ten(part[0:-1], base)
                if error2:
                    return res, None
                error, result = to_binary(res, data_type)
                if error:
                    return result, None
                parts[i] = result

        elif part[-1] == 'h':
            base = "h"
            error2, res = to_ten(part[0:-1], base)
            if error2:
                return res, None
            error, result = to_binary(res, data_type)
            if not error:
                parts[i] = result
            else:
                return result, None
        i += 1

    return None, parts


def parse_data(data_seg):
    data = []
    address = 0
    for line in data_seg:
        entry = {}

        parts = line.split()
        if len(parts) < 3:
            continue
        entry["address"] = address
        entry["name"] = parts[0]
        entry["type"] = data_types.get(parts[1].upper(), None)

        if entry["type"] is None:
            print(f"Unknown data type: {parts[1]}")
            continue

        error_message, values = parse_values(" ".join(parts[2:]), entry["type"])
        if error_message is None:
            entry["values"] = values
            data.append(entry)
        else:
            return error_message, None

        address += (entry["type"] // 8) * len(entry["values"])

    return None, data


def parse_label(line, address):
    parts = line.split(maxsplit=1)
    if len(parts) == 1 and parts[0][-1] == ':':
        return {"label": parts[0][:-1], "address": address, "name": None, "operands": None}
    pass


def contains_alphabet(s):
    return any(char.isalpha() for char in s) and not s[0:-1].isdigit()


def parse_instruction(instruction, address, data):
    # Split the instruction into its components
    parts = instruction.split(maxsplit=1)

    label = None
    if parts[0][-1] == ":":
        label = parts[0][:-1]
        parts2 = parts[1].split(maxsplit=1)
        name = parts2[0].upper()
        operands_str = parts2[1]
        # Split the operands by comma and strip any extra whitespace
        operands = [operand.strip() for operand in operands_str.split(",")]

        i = 0
        for operand in operands:
            if not is_number(operand) and operand.upper() in registers.keys():
                operands[i] = operand.upper()
            if i == 1:
                temp = operands[1].split(maxsplit=1)
                if temp[0].upper() == "LENGTHOF":
                    index = get_name_index(data, temp[1])
                    res = 0
                    for value in data[index]["values"]:
                        if contains_alphabet(value):
                            res += len(value) - 2
                        else:
                            res += 1
                    operands[i] = str(res)
                elif temp[0].upper() == "SIZEOF":
                    index = get_name_index(data, temp[1])
                    res = 0
                    for value in data[index]["values"]:
                        if contains_alphabet(value):
                            res += len(value) - 2
                        else:
                            res += 1

                    operands[i] = str(data[index]["type"] * res)
            i += 1

    else:
        name = parts[0].upper()
        if len(parts) > 1:
            operands_str = parts[1]
            # Split the operands by comma and strip any extra whitespace
            operands = [operand.strip() for operand in operands_str.split(",")]

            i = 0
            for operand in operands:
                if not is_number(operand) and operand.upper() in registers.keys():
                    operands[i] = operand.upper()
                if i == 1:
                    temp = operands[1].split(maxsplit=1)
                    if temp[0].upper() == "LENGTHOF":
                        index = get_name_index(data, temp[1])
                        res = 0
                        for value in data[index]["values"]:
                            if contains_alphabet(value):
                                res += len(value) -2
                            else:
                                res += 1
                        operands[i] = str(res)
                    elif temp[0].upper() == "SIZEOF":
                        index = get_name_index(data, temp[1])
                        res = 0
                        for value in data[index]["values"]:
                            if contains_alphabet(value):
                                res += len(value) - 2
                            else:
                                res += 1
                        operands[i] = str(data[index]["type"] * res)
                i += 1
        else:
            operands = None

    # Create the entry dictionary
    entry = {
        "label": label,
        "address": address,
        "name": name.upper(),
        "operands": operands
    }
    return entry


def check_name_exists(data_list, name_to_check):
    return any(entry['name'] == name_to_check for entry in data_list)


def get_name_index(data_list, name_to_check):
    for index, entry in enumerate(data_list):
        if entry['name'] == name_to_check:
            return index
    return -1  # Return -1 if the name is not found


def is_number(value):
    return isinstance(value, (int, float))


def parse_code(code_seg, data):
    address = 0
    code = []
    for line in code_seg:
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if parts[0][-1] == ":" and len(parts) == 1:  # code label can be on a line by itself
            entry = parse_label(line, address)
        else:
            entry = parse_instruction(line, address, data)
        if not entry["name"] and not entry["label"]:
            continue

        if entry["name"] is not None:
            if entry["name"] not in instructions.keys():
                error = f"Unknown instruction at line {address}"
                return error, None
            if entry["operands"] is not None and len(entry["operands"]) != instructions[entry["name"]][
                "operandsNumber"]:
                error = f"Operands number in line {address + 1} is wrong!"
                return error, None
            if entry["operands"] is not None and len(entry["operands"]) > 0 and is_number(entry["operands"][0]) and \
                    entry["name"] != "PUSH":
                error = f"Incorrect instruction use in line {address + 1}!"
                return error, None
            if instructions[entry["name"]]["same_size"]:
                op1 = entry["operands"][0]
                op2 = entry["operands"][1] if len(entry["operands"]) > 1 else None

                if (op1 in registers.keys()) and (op2 in registers.keys()):
                    if registers[op1][0] != registers[op2][0]:
                        error = f"Register size mismatch in line {address + 1}!"
                        return error, None

                if (op1 in registers.keys()) and (check_name_exists(data, op2)):
                    if registers[op1][0] != data[get_name_index(data, op2)]["type"]:
                        error = f"Operands size mismatch in line {address + 1}!"
                        return error, None

                if (op2 in registers) and (check_name_exists(data, op1)):
                    if registers[op2][0] != data[get_name_index(data, op1)]["type"]:
                        error = f"Operands size mismatch in line {address + 1}!"
                        return error, None

        code.append(entry)
        address += 1
    return "", code


def carry(result, bits):
    err, res = to_ten(result, 2)
    if res < 0 or res > 2 ** bits - 1:
        return 1
    return 0


def binary_addition(s, dest, bits):
    # Convert the binary strings to decimal integers
    source_decimal = int(s, 2)
    dest_decimal = int(dest, 2)

    # Perform the addition in decimal
    result_decimal = source_decimal + dest_decimal

    # Convert the result back to binary
    result_binary = bin(result_decimal)[2:]

    # Ensure the result fits within the specified capacity

    result_binary = result_binary.zfill(bits)

    return result_binary


def overflow(source, dest, result):
    if source[0] == dest[0] == 0 and result[0] == 1:
        return 1
    if source[0] == dest[0] == 1 and result[0] == 0:
        return 1
    return 0


def auxiliary(a, b, operation):
    ac_flag = False

    a_dec = binary_to_decimal(a)
    b_dec = binary_to_decimal(b)

    # Ensure a and b are within 8-bit range
    a_dec = a_dec & 0xFF
    b_dec = b_dec & 0xFF

    a_lower_nibble = a_dec & 0x0F
    b_lower_nibble = b_dec & 0x0F

    if operation == 'add':
        ac_flag = (a_lower_nibble + b_lower_nibble) > 0x0F
    elif operation == 'sub':
        ac_flag = (b_lower_nibble - a_lower_nibble) < 0
    elif operation == 'neg':
        # if the lower nibble of the original value is non-zero -> ac_flag is set
        ac_flag = (a_lower_nibble != 0)

    ac_flag = 1 if ac_flag else 0

    return ac_flag


def zero(result):
    if int(result) == 0:
        return 1
    return 0


def sign(result):
    if int(result[0]) == 1:
        return 1
    return 0


def parity(result, bits):
    if bits == 8:
        return 1 if result.count("1") % 2 == 0 else 0
    if bits == 16:
        return 1 if result[8:].count("1") % 2 == 0 else 0
    if bits == 32:
        return 1 if result[24:].count("1") % 2 == 0 else 0


def parse_op_dest(op, data):
    is_reg = is_mem = is_imm = False
    capacity = -1
    result = None
    if op in registers.keys():
        is_reg = True
        if registers[op][0] == 8:
            capacity = 8
            reg = registers[op][1]
            if op[-1] == 'H':
                result = hex_to_binary(REGISTERS[reg][4:6]).zfill(8)
            else:
                result = hex_to_binary(REGISTERS[reg][6:]).zfill(8)
        elif registers[op][0] == 16:
            capacity = 16
            reg = registers[op][1]
            result = hex_to_binary(REGISTERS[reg][4:]).zfill(16)

        else:  # 32 bit
            result = hex_to_binary(REGISTERS[op]).zfill(32)
            capacity = 32

    elif check_name_exists(data, op):
        is_mem = True
        index = get_name_index(data, op)
        result = data[index]["values"][0]
        capacity = data[index]["type"]

    else:
        is_imm = True

    return result, is_reg, is_imm, is_mem, capacity


def parse_op_source(op, data, capacity):
    result = None
    if op in registers.keys():
        if registers[op][0] == 8:
            reg = registers[op][1]
            if op[-1] == 'H':
                result = hex_to_binary(REGISTERS[reg][4:6]).zfill(8)
            else:
                result = hex_to_binary(REGISTERS[reg][6:]).zfill(8)
        elif registers[op][0] == 16:
            reg = registers[op][1]
            result = hex_to_binary(REGISTERS[reg][4:]).zfill(16)
        else:  # 32 bit
            result = hex_to_binary(REGISTERS[op]).zfill(32)

    elif check_name_exists(data, op):
        index = get_name_index(data, op)
        result = data[index]["values"][0]
    else:  # immediate (hex decimal(negative or positive) binary octal)
        if op.isdigit() or op[1:].isdigit():
            error, result = to_binary(op, capacity)
        elif op[0:-1].isdigit() and op[-1] != 'h':
            base = op[-1]
            if base == 'o' or base == 'q' or base == 'O' or base == 'Q':
                error2, res = to_ten(op[0:-1], 'o')
                error, result = to_binary(res, capacity)
            elif base == 'b':
                result = op[0:-1]
                result = result.zfill(capacity)

        elif op[-1] == 'h':
            base = "h"
            error2, res = to_ten(op[0:-1], base)
            error, result = to_binary(res, capacity)

    return result


def add(operands, data):
    op1 = operands[0]
    op2 = operands[1]

    dest, is_reg, is_imm, is_mem, capacity = parse_op_dest(op1, data)
    if is_imm:
        return "immediate operand not allowed as destination!", None
    if is_mem:
        if check_name_exists(data, op2):
            return "Memory to Memory in 'add' instruction is invalid!", None
    source = parse_op_source(op2, data, capacity)

    if dest is None:
        return source, None

    index = get_name_index(data, op1)

    result = binary_addition(source, dest, capacity)

    FLAGS["carry"] = carry(result, capacity)
    FLAGS["auxiliary"] = auxiliary(source, dest, "add")

    if len(result) > capacity:
        # Truncate to the least significant bits
        result = result[-capacity:]
    if is_reg:
        if registers[op1][0] == 8:
            reg = registers[op1][1]

            if registers[op1][-1] == 'H':
                temp = REGISTERS[reg][0:4]
                temp += binary_to_hex(result)
                temp += REGISTERS[reg][6:]
                REGISTERS[reg] = temp
            else:
                temp = REGISTERS[reg][0:6]
                temp += binary_to_hex(result)
                REGISTERS[reg] = temp
        elif registers[op1][0] == 16:
            reg = registers[op1][1]
            temp = REGISTERS[reg][0:4]
            temp += binary_to_hex(result)
            REGISTERS[reg] = temp
        else:
            REGISTERS[op1] = binary_to_hex(result)
    else:
        data[index]["values"][0] = result

    FLAGS["zero"] = zero(result)
    FLAGS["parity"] = parity(result, capacity)
    FLAGS["sign"] = sign(result)
    FLAGS["overflow"] = overflow(source, dest, result)

    return None, data


def sub(operands, data):
    op1 = operands[0]
    op2 = operands[1]

    dest, is_reg, is_imm, is_mem, capacity = parse_op_dest(op1, data)
    if is_imm:
        return "Immediate operand not allowed as destination!", None
    if is_mem:
        if check_name_exists(data, op2):
            return "Memory to Memory in 'sub' instruction is invalid!", None
    source = parse_op_source(op2, data, capacity)

    if dest is None:
        return source, None

    FLAGS["auxiliary"] = auxiliary(source, dest, "sub")

    # Carry Flag
    carry_flag = 0
    dest = binary_to_decimal(dest)
    source = binary_to_decimal(source)
    if isinstance(dest, str) or isinstance(source, str):
        return "Error: Invalid binary input", None
    res = dest - source

    if res < 0 or res > 2 ** capacity - 1:
        carry_flag = 1
    FLAGS["carry"] = carry_flag

    index = get_name_index(data, op1)

    error, result = to_binary(str(res), capacity)

    if len(result) > capacity:
        # Truncate to the least significant bits
        result = result[-capacity:]

    if is_reg:
        if registers[op1][0] == 8:
            reg = registers[op1][1]
            if registers[op1][-1] == 'H':
                temp = REGISTERS[reg][0:4] + binary_to_hex(result) + REGISTERS[reg][6:]
                REGISTERS[reg] = temp
            else:
                temp = REGISTERS[reg][0:6] + binary_to_hex(result)
                REGISTERS[reg] = temp
        elif registers[op1][0] == 16:
            reg = registers[op1][1]
            temp = REGISTERS[reg][0:4] + binary_to_hex(result)
            REGISTERS[reg] = temp
        else:
            REGISTERS[op1] = binary_to_hex(result)
    else:
        data[index]["values"][0] = result

    FLAGS["zero"] = zero(result)
    FLAGS["parity"] = parity(result, capacity)
    FLAGS["sign"] = sign(result)
    FLAGS["overflow"] = overflow(str(source), str(dest), result)

    return None, data


def mov(operands, data):
    op1 = operands[0]
    op2 = operands[1]

    dest, is_reg, is_imm, is_mem, capacity = parse_op_dest(op1, data)
    if is_imm:
        return "Immediate operand not allowed as destination!", None
    if is_mem:
        if check_name_exists(data, op2):
            return "Memory to Memory in 'sub' instruction is invalid!", None
    result = parse_op_source(op2, data, capacity)

    index = get_name_index(data, op1)

    if is_reg:
        if registers[op1][0] == 8:
            reg = registers[op1][1]
            if op1[-1] == 'H':
                temp = REGISTERS[reg][0:4] + binary_to_hex(result) + REGISTERS[reg][6:]
                REGISTERS[reg] = temp
            else:
                temp = REGISTERS[reg][0:6] + binary_to_hex(result)
                REGISTERS[reg] = temp
        elif registers[op1][0] == 16:
            reg = registers[op1][1]
            temp = REGISTERS[reg][0:4] + binary_to_hex(result)
            REGISTERS[reg] = temp
        else:
            REGISTERS[op1] = binary_to_hex(result)
    else:
        data[index]["values"][0] = result

    return None, data


def extended_mov(operands, data, signed):
    op1 = operands[0]
    op2 = operands[1]

    dest, is_reg, is_imm, is_mem, capacity = parse_op_dest(op1, data)
    if is_imm or is_mem:
        return "Immediate operand not allowed as destination!", None
    if registers[op1][0] == 8:
        return "byte register cannot be first operand!", None
    if (op2 not in registers.keys()) and (not check_name_exists(data, op2)):
        return "source operand can not be immediate!", None
    size = registers[op1][0]

    if op2 in registers.keys():
        if registers[op2][0] >= size:
            return "Invalid instruction operand!", None
    else:
        index = get_name_index(data, op2)
        if data[index]["type"] >= size:
            return "Invalid instruction operand!", None

    source = parse_op_source(op2, data, capacity)

    if source[0] == '0' or not signed:
        result = source.zfill(capacity)
    else:
        result = source.rjust(capacity, '1')

    if registers[op1][0] == 16:
        reg = registers[op1][1]
        temp = REGISTERS[reg][0:4] + binary_to_hex(result)
        REGISTERS[reg] = temp
    else:
        REGISTERS[op1] = binary_to_hex(result)

    return None, data


def xchg(operands, data):
    op1 = operands[0]
    op2 = operands[1]

    dest, is_reg, is_imm, is_mem, capacity = parse_op_dest(op1, data)
    source, is_reg2, is_imm2, is_mem2, capacity2 = parse_op_dest(op2, data)

    if is_imm or is_imm2:
        return "Immediate operands not allowed in 'XCHG' instruction!", None
    if is_mem and is_mem2:
        return "'XCHG' can not accept two memory operands!", None

    dest = dest.zfill(capacity)
    source = source.zfill(capacity2)
    temp2 = dest
    if is_reg:
        if registers[op1][0] == 8:
            reg = registers[op1][1]
            if op1[-1] == 'H':
                temp = REGISTERS[reg][0:4] + binary_to_hex(source) + REGISTERS[reg][6:]
                REGISTERS[reg] = temp
            else:
                temp = REGISTERS[reg][0:6] + binary_to_hex(source)
                REGISTERS[reg] = temp
        elif registers[op1][0] == 16:
            reg = registers[op1][1]
            temp = REGISTERS[reg][0:4] + binary_to_hex(source)
            REGISTERS[reg] = temp
        else:
            REGISTERS[op1] = binary_to_hex(source)
    elif is_mem:
        i = get_name_index(data, op1)
        data[i]["values"][0] = source

    if is_reg2:
        if registers[op2][0] == 8:
            reg = registers[op2][1]
            if op2[-1] == 'H':
                temp = REGISTERS[reg][0:4] + binary_to_hex(temp2) + REGISTERS[reg][6:]
                REGISTERS[reg] = temp
            else:
                temp = REGISTERS[reg][0:6] + binary_to_hex(temp2)
                REGISTERS[reg] = temp
        elif registers[op2][0] == 16:
            reg = registers[op2][1]
            temp = REGISTERS[reg][0:4] + binary_to_hex(temp2)
            REGISTERS[reg] = temp
        else:
            REGISTERS[op2] = binary_to_hex(temp2)
    elif is_mem:
        i = get_name_index(data, op1)
        data[i]["values"][0] = temp2

    return None, data


def inc(operands, data):
    arr = [operands[0], '1']
    carry_flag = FLAGS["carry"]
    error, data = add(arr, data)
    FLAGS["carry"] = carry_flag
    if data is None:
        return error, None
    return None, data


def dec(operands, data):
    arr = [operands[0], '1']
    carry_flag = FLAGS["carry"]
    error, data = sub(arr, data)
    FLAGS["carry"] = carry_flag
    if data is None:
        return error, None
    return None, data


def neg(operands, data):
    def binary_to_h(binary_str):
        return hex(int(binary_str, 2))[2:].upper().zfill(len(binary_str) // 4)

    op = operands[0]
    res, is_reg, is_imm, is_mem, capacity = parse_op_dest(op, data)
    if is_imm:
        return "Immediate is not allowed for 'NEG' instruction!", None

    res = res.zfill(capacity)
    twos_complement = bin((1 << capacity) - int(res, 2))[2:]
    twos_complement = twos_complement.zfill(capacity)
    if len(twos_complement) > capacity:
        # Truncate to the least significant bits
        twos_complement = twos_complement[-capacity:]

    if is_reg:
        reg = registers[op][1]
        if registers[op][0] == 8:
            if op[-1] == 'H':
                REGISTERS[reg] = REGISTERS[reg][0:4] + binary_to_h(twos_complement) + REGISTERS[reg][6:]
            else:
                REGISTERS[reg] = REGISTERS[reg][0:6] + binary_to_h(twos_complement)
        elif registers[op][0] == 16:
            REGISTERS[reg] = REGISTERS[reg][0:4] + binary_to_h(twos_complement)
        else:
            REGISTERS[op] = binary_to_h(twos_complement)
    else:
        data[get_name_index(data, op)]["values"][0] = twos_complement

    FLAGS["auxiliary"] = auxiliary(twos_complement, twos_complement, "neg")
    FLAGS["zero"] = zero(twos_complement)
    FLAGS["carry"] = 1 if not FLAGS["zero"] else 0
    FLAGS["parity"] = parity(twos_complement, capacity)
    FLAGS["sign"] = sign(twos_complement)
    FLAGS["overflow"] = overflow(res, res, twos_complement)

    return None, data


def bitwise_instruction(operands, data, is_and, is_or):
    op1 = operands[0]
    op2 = operands[1]

    dest, is_reg, is_imm, is_mem, capacity = parse_op_dest(op1, data)
    if is_imm:
        return "immediate operand not allowed as destination!", None

    source = parse_op_source(op2, data, capacity)

    index = get_name_index(data, op1)
    if is_and:
        result = bin(int(dest, 2) & int(source, 2))[2:].zfill(len(dest))
        FLAGS["auxiliary"] = auxiliary(source, dest, "and")

    elif is_or:
        result = bin(int(dest, 2) | int(source, 2))[2:].zfill(len(dest))
        FLAGS["auxiliary"] = auxiliary(source, dest, "or")

    else:
        result = bin(int(dest, 2) ^ int(source, 2))[2:].zfill(len(dest))
        FLAGS["auxiliary"] = auxiliary(source, dest, "xor")

    if len(result) > capacity:
        # Truncate to the least significant bits
        result = result[-capacity:]
    if is_reg:
        if registers[op1][0] == 8:
            reg = registers[op1][1]

            if registers[op1][-1] == 'H':
                temp = REGISTERS[reg][0:4]
                temp += binary_to_hex(result)
                temp += REGISTERS[reg][6:]
                REGISTERS[reg] = temp
            else:
                temp = REGISTERS[reg][0:6]
                temp += binary_to_hex(result)
                REGISTERS[reg] = temp
        elif registers[op1][0] == 16:
            reg = registers[op1][1]
            temp = REGISTERS[reg][0:4]
            temp += binary_to_hex(result)
            REGISTERS[reg] = temp
        else:
            REGISTERS[op1] = binary_to_hex(result)
    else:
        data[index]["values"][0] = result

    FLAGS["zero"] = zero(result)
    FLAGS["parity"] = parity(result, capacity)
    FLAGS["sign"] = sign(result)
    FLAGS["overflow"] = 0
    FLAGS["carry"] = 0

    return None, data


def compare(operands, data):
    global FLAGS, REGISTERS

    # Create deep copies of data and REGISTERS
    regs_copy = copy.deepcopy(REGISTERS)
    data_temp = copy.deepcopy(data)

    # Perform the subtraction operation
    error, data = sub(operands, data)

    # Restore the original data and REGISTERS
    data = data_temp
    REGISTERS = regs_copy

    return error, data


def decimal_to_hex(decimal_int, capacity):
    # Convert the decimal integer to a hexadecimal string and remove the '0x' prefix
    hex_str = hex(decimal_int)[2:]

    # Ensure the hexadecimal string has the required capacity (minimum length)
    hex_str = hex_str.zfill(capacity)

    return hex_str


def push(operands, data):
    global top
    global runtime_stack
    op1 = operands[0]
    result, is_reg, is_imm, is_mem, capacity = parse_op_dest(op1, data)
    if capacity == 8:
        return "instruction formats for push is not correct!", None
    if is_imm and capacity != 32:
        return "instruction formats for push is not correct!", None
    if top == "0000":
        return "stack is full!", None
    value = binary_to_hex(result)
    entry = {
        "address": top,
        "value": value.zfill(capacity)
    }
    
    err, res2 = (to_ten(top, 'h'))
    top = decimal_to_hex(res2 - capacity // 8, capacity // 8)
    
    runtime_stack.append(entry)
    return None, data


def pop(operands, data):
    global top
    global runtime_stack
    op1 = operands[0]
    result, is_reg, is_imm, is_mem, capacity = parse_op_dest(op1, data)
    if capacity == 8:
        return "instruction formats for pop is not correct!", None
    if is_imm:
        return "instruction formats for push is not correct!", None
    if top == "FFFF":
        return "stack is empty!", None

    popped_elem = runtime_stack[top]
    if is_reg:
        if capacity == 16:
            temp_reg = registers[op1][1]
            REGISTERS[temp_reg] = REGISTERS[temp_reg][:4] + popped_elem.zfill(capacity)
        else:
            REGISTERS[op1] = popped_elem.zfill(capacity)
    if is_mem:
        index = get_name_index(data, op1)
        err, temp = hex_to_binary(popped_elem).zfill(capacity)
        data[index]["values"][0] = temp

    err, res2 = (to_ten(top, 'h'))
    top = decimal_to_hex(res2 + capacity // 8, capacity // 8)

    return None, data


def loop(line, code):
    is_loop = is_taken = False
    jumped_line = 0

    if line["name"].upper() == 'LOOP':
        is_loop = True
        if REGISTERS["ECX"] != "00000000":
            is_taken = True

            # ECX = ECX - 1
            error, temp = to_ten(REGISTERS["ECX"], 'h')
            error, temp = to_binary(str(temp - 1), 32)
            temp = binary_to_hex(temp)
            REGISTERS["ECX"] = temp

            for elem in code:
                if elem["label"] == line["label"]:
                    return is_loop, is_taken, jumped_line
                jumped_line += 1

    return is_loop, is_taken, jumped_line


def perform(line, data):
    data2 = []
    if line["name"] == "ADD":
        error_mes, data2 = add(line["operands"], data)
        if error_mes is not None:
            return error_mes, None
    elif line["name"] == "SUB":
        error_mes, data2 = sub(line["operands"], data)
        if error_mes is not None:
            return error_mes, None
    elif line["name"] == "MOV":
        error_mes, data2 = mov(line["operands"], data)
        if error_mes is not None:
            return error_mes, None
    elif line["name"] == "MOVSX" or line["name"] == "MOVZX":
        error_mes, data2 = extended_mov(line["operands"], data, line["name"] == "MOVSX")
        if error_mes is not None:
            return error_mes, None
    elif line["name"] == "XCHG":
        error_mes, data2 = xchg(line["operands"], data)
        if error_mes is not None:
            return error_mes, None
    elif line["name"] == "INC":
        error_mes, data2 = inc(line["operands"], data)
        if error_mes is not None:
            return error_mes, None
    elif line["name"] == "DEC":
        error_mes, data2 = dec(line["operands"], data)
        if error_mes is not None:
            return error_mes, None
    elif line["name"] == "NEG":
        error_mes, data2 = neg(line["operands"], data)
        if error_mes is not None:
            return error_mes, None
    elif line["name"] == "AND" or line["name"] == "OR" or line["name"] == "XOR":
        error_mes, data2 = bitwise_instruction(line["operands"], data, line["name"] == "AND", line["name"] == "OR")
        if error_mes is not None:
            return error_mes, None
    elif line["name"] == "PUSH":
        error_mes, data2 = push(line["operands"], data)
        if error_mes is not None:
            return error_mes, None
    elif line["name"] == "POP":
        error_mes, data2 = pop(line["operands"], data)
        if error_mes is not None:
            return error_mes, None
    elif line["name"] == "CMP":
        error_mes, data2 = compare(line["operands"], data)
        if error_mes is not None:
            return error_mes, None

    return None, data2


def memory_change(data):
    memory = []  # address = index
    for elem in data:
        for value in elem["values"]:
            if value.isdigit():
                hex_val = binary_to_hex(value)
                end = len(hex_val) - 1
                start = end - 1
                while start >= 0:
                    memory.append(hex_val[start:end + 1])
                    start -= 2
                    end -= 2
            else:
                for i in range(len(value)):
                    if i == 0 or i == len(value) - 1:
                        continue
                    ascii_value = ord(value[i])
                    error, val = to_binary(str(ascii_value), 8)
                    hex_val = binary_to_hex(val)
                    end = len(hex_val) - 1
                    start = end - 1
                    while start >= 0:
                        memory.append(hex_val[start:end + 1])
                        start -= 2
                        end -= 2
    return memory


def search_for_label(label, code):
    index = -1
    for elem in code:
        if elem["label"] == label:
            return elem["address"]

    return index


def jump(line, code, left_op, right_op):
    is_jmp = True
    is_taken = True
    i = -1

    # Jumps Based on Specific Flags
    if line["name"].upper() == "JMP":
        i = search_for_label(line["operands"][0], code)
    elif line["name"].upper() == "JZ":
        if FLAGS["zero"] == 1:
            is_taken = True
            i = search_for_label(line["operands"][0], code)
    elif line["name"].upper() == "JNZ":
        if FLAGS["zero"] == 0:
            is_taken = True
            i = search_for_label(line["operands"][0], code)
    elif line["name"].upper() == "JC":
        if FLAGS["carry"] == 1:
            is_taken = True
            i = search_for_label(line["operands"][0], code)
    elif line["name"].upper() == "JNC":
        if FLAGS["carry"] == 0:
            is_taken = True
            i = search_for_label(line["operands"][0], code)
    elif line["name"].upper() == "JO":
        if FLAGS["overflow"] == 1:
            is_taken = True
            i = search_for_label(line["operands"][0], code)
    elif line["name"].upper() == "JNO":
        if FLAGS["overflow"] == 0:
            is_taken = True
            i = search_for_label(line["operands"][0], code)
    elif line["name"].upper() == "JS":
        if FLAGS["sign"] == 1:
            is_taken = True
            i = search_for_label(line["operands"][0], code)
    elif line["name"].upper() == "JNS":
        if FLAGS["sign"] == 0:
            is_taken = True
            i = search_for_label(line["operands"][0], code)
    elif line["name"].upper() == "JP":
        if FLAGS["parity"] == 1:
            is_taken = True
            i = search_for_label(line["operands"][0], code)
    elif line["name"].upper() == "JNP":
        if FLAGS["parity"] == 1:
            is_taken = True
            i = search_for_label(line["operands"][0], code)

    # Jumps Based on Equality
    elif line["name"].upper() == "JE":
        if left_op == right_op:
            is_taken = True
            i = search_for_label(line["operands"][0], code)
    elif line["name"].upper() == "JNE":
        if left_op != right_op:
            is_taken = True
            i = search_for_label(line["operands"][0], code)
    elif line["name"].upper() == "JCXZ":
        if REGISTERS["ECX"][4:] == "0000":
            is_taken = True
            i = search_for_label(line["operands"][0], code)
    elif line["name"].upper() == "JECXZ":
        if REGISTERS["ECX"] == "00000000":
            is_taken = True
            i = search_for_label(line["operands"][0], code)

    # Jumps Based on Unsigned Comparisons
    elif line["name"].upper() == "JA" or line["name"].upper() == "JNBE":
        if left_op > right_op:
            is_taken = True
            i = search_for_label(line["operands"][0], code)
    elif line["name"].upper() == "JAE" or line["name"].upper() == "JNB":
        if left_op >= right_op:
            is_taken = True
            i = search_for_label(line["operands"][0], code)
    elif line["name"].upper() == "JB" or line["name"].upper() == "JNAE":
        if left_op < right_op:
            is_taken = True
            i = search_for_label(line["operands"][0], code)
    elif line["name"].upper() == "JBE" or line["name"].upper() == "JNA":
        if left_op <= right_op:
            is_taken = True
            i = search_for_label(line["operands"][0], code)

    # Jumps Based on Signed Comparisons
    elif line["name"].upper() == "JG" or line["name"].upper() == "JNLE":
        if left_op > right_op:
            is_taken = True
            i = search_for_label(line["operands"][0], code)
    elif line["name"].upper() == "JGE" or line["name"].upper() == "JNL":
        if left_op >= right_op:
            is_taken = True
            i = search_for_label(line["operands"][0], code)
    elif line["name"].upper() == "JL" or line["name"].upper() == "JNGE":
        if left_op < right_op:
            is_taken = True
            i = search_for_label(line["operands"][0], code)
    elif line["name"].upper() == "JLE" or line["name"].upper() == "JNG":
        if left_op <= right_op:
            is_taken = True
            i = search_for_label(line["operands"][0], code)

    if i == -1:
        is_jmp = False
        is_taken = False
        i = None

    return is_jmp, is_taken, i


def print_stack(arr):
    for element in arr:
        print(element)
    pass


def main():
    print("Enter 1 to read from a file, or 2 otherwise:")
    option = int(input().strip())
    lines = []

    if option == 1:
        print("Enter the filename:")
        filename = input().strip()
        try:
            with open(filename, 'r') as file:
                lines = file.readlines()
        except FileNotFoundError:
            print(f"File {filename} not found.")
            return
    else:
        print("Enter lines of input (blank line to finish):")
        while True:
            line = input().strip()
            if line == "":  # Break the loop if the input is an empty line
                break
            lines.append(line)

    is_data = False
    is_code = False
    data_seg = []
    code_seg = []

    # Process each line to creat data segment and code segment array
    for line in lines:
        if ".data" in line and not is_data:
            is_data = True
            is_code = False
        elif ".code" in line and not is_code:
            is_code = True
            is_data = False
        elif is_code:
            code_seg.append(line)
        elif is_data:
            data_seg.append(line)

    error, data = parse_data(data_seg)
    if data is None:
        print(error)
        return
    error, code = parse_code(code_seg, data)
    if error != "":
        print(error)
        return

    lines = []
    memory = []
    left_op = right_op = None

    i = 0
    line = code[i]

    while i < len(code):
        # is_taken -> shows if a jump has happened!
        is_jump, is_taken, jumped_line = jump(line, code, left_op, right_op)
        if is_jump and is_taken:
            i = jumped_line
        # is_taken -> if ECX > 0
        is_loop, is_taken, jumped_line = loop(line, code)
        if is_loop and is_taken:
            i = jumped_line

        line = code[i]
        if line["label"] is None or line["name"] is not None:
            if line["name"].upper() == "CMP":
                left_op, is_reg, is_imm, is_mem, capacity = parse_op_dest(line["operands"][0], data)
                right_op = parse_op_source(line["operands"][1], data, capacity)
            error, data = perform(line, data)
            memory = memory_change(data)
            if data is None:
                print(error)
                return
            lines.append(
                {"line_num": line["address"] + 1, "regs": copy.deepcopy(REGISTERS), "flags": copy.deepcopy(FLAGS),
                 "memory": memory, "stack": copy.deepcopy(runtime_stack)}
            )
            i += 1
        else:
            lines.append(
                {"line_num": line["address"] + 1, "regs": copy.deepcopy(REGISTERS), "flags": copy.deepcopy(FLAGS),
                 "memory": memory, "stack": copy.deepcopy(runtime_stack)}
            )
            i += 1

    while True:
        print("1: Print Flags\n2: Print Registers\n3: Show stack\n4: Show memory\n5: Exit")
        option = int(input().strip())

        if option == 1:
            print("1: See flags after a specific line\n2: See final flags")
            option2 = int(input().strip())
            if option2 == 1:
                seen = False
                print("Enter your desired line")
                line_num = int(input().strip())
                for line in lines:
                    if line["line_num"] == line_num and not seen:
                        print(line["flags"])
                        seen = True
                        continue
                    if seen:
                        print("1: Continue\t2: Exit")
                        option3 = int(input().strip())
                        if option3 == 1:
                            print(line["flags"])
                        else:
                            break

            else:
                print(lines[len(lines) - 1]["flags"])

        elif option == 2:
            print("1: See registers after a specific\n2: See final registers")
            option2 = int(input().strip())
            if option2 == 1:
                seen = False
                print("Enter your desired line")
                line_num = int(input().strip())
                for line in lines:
                    if line["line_num"] == line_num and not seen:
                        print(line["regs"])
                        seen = True
                        continue
                    if seen:
                        print("1: Continue\t2: Exit")
                        option3 = int(input().strip())
                        if option3 == 1:
                            print(line["regs"])
                        else:
                            break

            else:
                print(lines[len(lines) - 1]["regs"])

        elif option == 3:
            print("1: See stack after a specific line\n2: See final stack")
            option2 = int(input().strip())
            if option2 == 1:
                seen = False
                print("Enter your desired line")
                line_num = int(input().strip())
                for line in lines:
                    if line["line_num"] == line_num and not seen:
                        print_stack(line["stack"])
                        seen = True
                        continue
                    if seen:
                        print("1: Continue\t2: Exit")
                        option3 = int(input().strip())
                        if option3 == 1:
                            print_stack(line["stack"])
                        else:
                            break
            else:
                print_stack(lines[len(lines) - 1]["stack"])

        elif option == 4:
            print("1: See memory after a specific line\n2: See final memory")
            option2 = int(input().strip())
            if option2 == 1:
                seen = False
                print("Enter your desired line")
                line_num = int(input().strip())
                for line in lines:
                    if line["line_num"] == line_num and not seen:
                        print(line["memory"])
                        seen = True
                        continue
                    if seen:
                        print("1: Continue\t2: Exit")
                        option3 = int(input().strip())
                        if option3 == 1:
                            print(line["memory"])
                        else:
                            break
            else:
                print(lines[len(lines) - 1]["memory"])

        else:
            break

if __name__ == "__main__":
    main()
