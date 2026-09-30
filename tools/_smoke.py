"""Smoke test for downloader.py sandbox safety."""
import sys, os
sys.path.insert(0, r'D:\zhengjianxi\agent\MiniMaxCode\tools\jxzhen-webprojects\aec-2027-vibevoice-agent')
os.environ['SANDBOX'] = r'C:\Users\Administrator\AppData\Local\Temp\minimax_test'
from agent.downloader import assert_inside_sandbox

# Test 1: in-sandbox
try:
    assert_inside_sandbox(r'C:\Users\Administrator\AppData\Local\Temp\minimax_test\foo')
    print('in-sandbox: OK')
except SystemExit:
    print('in-sandbox: FAILED')

# Test 2: out-of-sandbox should exit
print('Testing escape...')
try:
    assert_inside_sandbox(r'C:\Windows\Temp\escape')
    print('ESCAPE: NOT BLOCKED (bad!)')
except SystemExit as e:
    print(f'ESCAPE: BLOCKED (good, exit={e.code})')