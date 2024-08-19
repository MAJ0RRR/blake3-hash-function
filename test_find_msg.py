import pytest
from util import hash, find_msg
import time

TEST_INPUTS = {
    2 : "D988C5249058DFA36BC5D1E219E05C2C",
    3 : "E54EE7FBCFFAF9ED15F35B0CBD32C8A1",
    4 : "E6D1F4D1BA7B97D1962D62E26B57A2A8",
    5 : "AC9971C304CF9F0BA513AF861F0341E8",
    6 : "E5F86BBA01F95D15187E28A3CE8C5C42",
    7 : "23B86B052AAEA5E54EA55FAA7DF4415C",
    8 : "21AF7ED1072AB3D3C497F7865C922338"
}

SYMBOLS = "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890!@#$%^&*-_=+([{<)]}>'\";:?,.\/|"

def test_find_msg():
    """
    Test the find message function by comparing found messasge hash with initial hash.

    :raises AssertionError: If the found hash does not match the computed hash
    """
    def find_msg_for_length(msg_length):
        expected_hash = TEST_INPUTS[msg_length]
        found_msg = find_msg(msg_length, expected_hash, SYMBOLS)
        hash_found_msg = hash(found_msg)
        hash_found_msg = ['{:04X}'.format(value) for value in hash_found_msg]
        hash_found_msg = ''.join(hash_found_msg)
        print(f"\n({msg_length}) Found message is: {found_msg}")
        print(f"\tExpected hash: {expected_hash}\n\tHash from hash function: {hash_found_msg}")
        assert hash_found_msg == expected_hash, f"Hash mismatch for length {msg_length}. Expected {expected_hash}, got {hash_found_msg}"

    start_time = time.time()
    for msg_length in TEST_INPUTS.keys():
        current_time = time.time()
        find_msg_for_length(msg_length)
        end_time = time.time()
        print(f"\tExecution time for length `{msg_length}` is {end_time - current_time} seconds.")
        print(f"\tTotal time so far is: {end_time - start_time} seconds.")
