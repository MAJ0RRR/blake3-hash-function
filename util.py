import logging
import itertools
import time
import multiprocessing
from multiprocessing import Manager

# Change this variable to toggle the logger on/off
DEBUG_MODE = False
INFO_MODE = True

# Configure logger
log_format = "%(levelname)s:\t%(message)s"
if DEBUG_MODE:
    logging.basicConfig(level=logging.DEBUG, format=log_format)
elif INFO_MODE:
    logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)

SYMBOLS = "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890!@#$%^&*-_=+([{<)]}>'\";:?,.\/|"

def make_v_matrix(w : list, block_n : int) -> list:
    """
    Create 4 x 4 v matrix

    :param w: 1D array with 8 elements, each being 16 bit number
    :param block_n: Number of currently processed block

    :return: 2D array of size 4 x 4
    
    """
    size = 4
    matrix = [[0] * size for _ in range(size)]

    for i in range(2):
        for j in range(size):
            matrix[i][j] = w[i * size + j]

    matrix[2][0] = 0x03F4
    matrix[2][1] = 0x774C 
    matrix[2][2] = 0x5690
    matrix[2][3] = 0xC878

    matrix[3][0] = matrix [3][2] = matrix[3][3] = 0x0000
    matrix[3][1] = block_n

    return matrix

def G(a : int, b: int, c: int, d: int, x: int, y: int) -> tuple[int, int, int, int]:
    """
    Helper function changing values of a,b,c,d

    :param a: Value from v matrix
    :param b: Value from v matrix
    :param c: Value from v matrix
    :param d: Value from v matrix
    :param x: Value from m list
    :param y: Value from m list

    :return: Tuple for new a,b,c,d values
    """
    a = (a + b + x) & 0xFFFF
    d = rol(d ^ a, 3)
    c = (c + d) & 0xFFFF
    b = rol(b ^ c, 11)
    a = (a + b + y) & 0xFFFF
    d = rol(d ^ a, 2)
    c = (c + d) & 0xFFFF
    b = rol(b ^ c, 5)
    
    return a, b, c, d

def rol(value: int, shift: int, bit_size: int = 16) -> int:
    """
    Perform cyclic rotate to the left

    :param value: 16 bit number
    :param shift: By how many bits should value be rotated
    :param bit_size: How big (in bytes) the value is

    :return: Rotated value
    """
    if not (0 <= value < 2**16):
        raise ValueError("Value must be 16-bit integers (0 <= value < 65536)")
    
    shift = shift % 16
    
    value &= (1 << bit_size) - 1
    return ((value << shift) & ((1 << bit_size) - 1)) | (value >> (bit_size - shift))

def transform(v : list, m: list) -> list:
    """
    Perform both vertical and diagonal transformations

    :param v: 2D array of size 4 x 4
    :param m: 1D array with 16 elements (current block), each being 16 bit number

    :return: 2D array of size 4 x 4 after transformations
    """

    # Verical transformations
    v[0][0], v[1][0], v[2][0], v[3][0] = G(v[0][0], v[1][0], v[2][0], v[3][0], m[0], m[1])
    v[0][1], v[1][1], v[2][1], v[3][1] = G(v[0][1], v[1][1], v[2][1], v[3][1], m[2], m[3])
    v[0][2], v[1][2], v[2][2], v[3][2] = G(v[0][2], v[1][2], v[2][2], v[3][2], m[4], m[5])
    v[0][3], v[1][3], v[2][3], v[3][3] = G(v[0][3], v[1][3], v[2][3], v[3][3], m[6], m[7])

    logger.debug(f"Pionowe przekształcenia: {['{:04X}'.format(value) for row in v for value in row]}")

    # Diagonal transformations
    v[0][0], v[1][1], v[2][2], v[3][3] = G(v[0][0], v[1][1], v[2][2], v[3][3], m[8], m[9])
    v[0][1], v[1][2], v[2][3], v[3][0] = G(v[0][1], v[1][2], v[2][3], v[3][0], m[10], m[11])
    v[0][2], v[1][3], v[2][0], v[3][1] = G(v[0][2], v[1][3], v[2][0], v[3][1], m[12], m[13])
    v[0][3], v[1][0], v[2][1], v[3][2] = G(v[0][3], v[1][0], v[2][1], v[3][2], m[14], m[15])

    logger.debug(f"Ukośne przekształcenia: {['{:04X}'.format(value) for row in v for value in row]}")

    return v

def permutate_m(m):
    """
    Perform specific permutation s on block

    :param m: 1D array with 16 elements (current block), each being 16 bit number
    
    :return: 1D array with 16 elements, being permutated block
    """

    s = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]

    p_m = [None] * len(m)
    
    for i, index in enumerate(s):
        p_m[index] = m[i]

    logger.debug(f"Dane m po permutacji {['{:04X}'.format(value) for value in p_m]}")
    
    return p_m

def msg_to_blocks(msg : str) -> list:
    """
    Transform message to blocks. Each block has 16 elements, each being 16 bit number so block size is 32 bytes
    Separate message into 32 bytes chunks. Bytes in each block are grouped into 16 x 16 bit numbers
    If last chunk has less than 32 bytes, complement it with n bytes of remaining value n
    If last chunk does not need a complement, add artificial chunk containing only complement values

    :param msg: String representation of the message

    :return: 2D array of blocks, where each element inside is block itself
    """

    # Empty message edge case, just return the complement
    if msg == "":
        return [[0x2020] * 16]
    
    byte_array = bytearray(msg, 'utf-8')

    def bytes_to_16bit_numbers(bytes_chunk: bytearray) -> list:
        """
        Helper function to transform bytearray to 1D array of 16 bit numbers

        :param bytes_chunk: Block in form of bytearray

        :return: 1D array of 16 bit numbers, being formatted block
        """
        return [int.from_bytes(bytes_chunk[i:i+2], 'big') for i in range(0, len(bytes_chunk), 2)]
    
    chunks = [byte_array[i:i+32] for i in range(0, len(byte_array), 32)]
    
    result = []

    for i,chunk in enumerate(chunks):
        if len(chunk) == 32:
            numbers = bytes_to_16bit_numbers(chunk)
            result.append(numbers)
            if i == len(chunks) - 1:
                result.append([0x2020] * 16)
        else:
            padding_length = 32 - len(chunk)
            padding_length_hex = format(padding_length, "02x")
            padding_num = int(f"0x{padding_length_hex}", 16)
            padded_chunk = chunk + bytearray([padding_num] * (padding_length))
            padded_numbers = bytes_to_16bit_numbers(padded_chunk)
            result.append(padded_numbers)
    
    return result

def hash(msg):
    """
    Main function which perform hashing of the message

    :param msg: String representation of the message

    :return: 1D array with 8 elements, each being 16 bit number which is hash value
    """
    # Convert message to blocks
    blocks = msg_to_blocks(msg)

    # Initialize state w as 8 x 16 bit numbers containing zeros
    w = [0 for _ in range(8)]

    for block_n, block in enumerate(blocks):
        # Initialize matrix v
        v = make_v_matrix(w,block_n)
        # Current block
        m = block

        logger.debug(f"Dane początkowe m: {['{:04X}'.format(value) for value in m]}")

        for r in range(6):
            logger.debug(f"Po {r} rundzie:")

            # Vertical and diagonal transformations
            v = transform(v, m)

            # Permuatation of m
            m = permutate_m(m)

            # Update state
            if r == 5:
                for i in range(4):
                    w[i] ^= v[0][i] ^ v[2][i]
                    w[i + 4] ^= v[1][i] ^ v[3][i]

    logger.debug(f"Funkcja skrótu {['{:04X}'.format(value) for value in w]}")

    return w

def hash_to_hex_string(h):
    return ''.join(['{:02X}'.format(b) for b in h])

def generate_combinations(symbols, length):
    for combination in itertools.product(symbols, repeat=length):
        yield ''.join(combination)

def worker(chunk, known_hash, found_flag, result_queue):
    for msg in chunk:
        if found_flag.value:  # Check if a match was already found
            return
        found_hash = hash(msg)
        found_hash = hash_to_hex_string(found_hash)
        if found_hash == known_hash:
            result_queue.put(msg)
            found_flag.value = 1  # Set the flag to indicate a match was found
            return

def find_msg(length: int, known_hash: str, symbols: str) -> str:
    start_time = time.time()

    num_cores = multiprocessing.cpu_count()
    logger.info(f"Found {num_cores} cores ont this machine")
    combinations = generate_combinations(symbols, length)
    chunk_size = len(symbols) ** length // num_cores
    logger.info(f"Chunk size is {chunk_size}")

    manager = Manager()
    result_queue = manager.Queue()
    found_flag = manager.Value('i', 0)  # Shared flag to indicate if a match is found
    processes = []

    for _ in range(num_cores):
        chunk = list(itertools.islice(combinations, chunk_size))
        if not chunk:
            break
        p = multiprocessing.Process(target=worker, args=(chunk, known_hash, found_flag, result_queue))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Check if we found a result
    if not result_queue.empty():
        end_time = time.time()
        logger.info(f"Execution time: {end_time - start_time} seconds")
        return result_queue.get()

    end_time = time.time()
    logger.info(f"Execution time: {end_time - start_time} seconds")
    return None