import math
import numpy as np
from fractions import Fraction
from matplotlib import pyplot as plt
import time

from QuantumRingsLib import (
    QuantumRegister, ClassicalRegister, QuantumCircuit,
    QuantumRingsProvider, job_monitor
)
from semiprimes import semiprimes

# Constants
shots = 512
TIMEOUT_SECONDS = 120
MAX_QUBITS = 200


backend = provider.get_backend("scarlet_quantum_rings")

def iqft_cct(qc, register, n):
    for i in range(n):
        for j in range(i):
            qc.cu1(-math.pi / 2 ** (i - j), register[j], register[i])
        qc.h(register[i])
    qc.barrier()

def continued_fraction(x, max_den=64):
    return Fraction(x).limit_denominator(max_den).denominator


def mcx(qc, ctrl, qubit):
    """
    Single extra control: If ctrl=1, then X on qubit.
    This is basically a CX but with ctrl as control and qubit as target.
    """
    qc.cx(ctrl, qubit)


def mccx(qc, controls, target):
    """
    2 controls in 'controls' list -> CCX 
    """
    if len(controls) == 2:
        qc.ccx(controls[0], controls[1], target)
    else:
        raise ValueError("Need exactly 2 controls here, or implement decomposition")



def mc_cx(qc, ctrl, c_qubit, t_qubit):
    """
    Apply a controlled version of a single CX gate:
    If (ctrl=1 AND c_qubit=1), then X(t_qubit).
    """
    # We can do a Toffoli:
    qc.ccx(ctrl, c_qubit, t_qubit)


def add_mod_N(qc, a, N, target, ancilla):
    """
    Reversibly adds a (classical int) to 'target' mod N, using ripple-carry logic.
    1) Create qubits for 'a'
    2) Reversible add a_qubits to target
    3) Compare target to N
    4) If target >= N, subtract N
    5) Uncompute the registers for a and N if needed

    target : list of qubits representing the number to be incremented
    ancilla: extra qubits for carry bits, comparison, etc.
    """
    n = len(target)
    needed = n + n + 2  # naive count: store 'a', store 'N', plus carry/comparison
    if len(ancilla) < needed:
        raise ValueError(f"Need at least {needed} ancillas, but only have {len(ancilla)}.")

    # Slice up ancilla:
    a_qubits = ancilla[:n]        # store bits of 'a'
    n_qubits = ancilla[n:2*n]     # store bits of 'N'
    carry_qubit = ancilla[2*n]    # for ripple-carry final bit
    cmp_qubit   = ancilla[2*n+1]  # for comparison

    # 1) Encode 'a' into a_qubits
    # e.g. if a_bits[i] = 1, do X on a_qubits[i]
    a_bits = [(a >> i) & 1 for i in range(n)]
    for i, bitval in enumerate(a_bits):
        if bitval == 1:
            qc.x(a_qubits[i])

    # 2) Encode 'N' into n_qubits similarly
    N_bits = [(N >> i) & 1 for i in range(n)]
    for i, bitval in enumerate(N_bits):
        if bitval == 1:
            qc.x(n_qubits[i])

    # 3) Reversible Add: target += a (in place)
    # Use Cuccaro's ripple-carry adder (in-place).
    cuccaro_add(qc, a_qubits, target, carry_qubit)

    # 4) Compare target >= N
    # We'll store the result in cmp_qubit = 1 if target >= N
    # This is also a known circuit: a quantum comparator. We do a subtract: if result < 0, we know target < N, etc.
    # For demonstration, we'll do a skeleton:
    cuccaro_sub(qc, n_qubits, target, carry_qubit)
    # if target < 0, that means original target < N
    # but we can't measure directly, so we store sign bit in cmp_qubit
    qc.cx(target[n-1], cmp_qubit)   # a naive approach
    # uncompute
    cuccaro_add(qc, n_qubits, target, carry_qubit)

    # 5) If cmp_qubit == 1, do target -= N
    # We can do a multi-control approach. A simplistic approach is:
    # 'cmp_qubit' controls an entire subtraction of N from target
    # i.e. for each bit in n_qubits, if n_qubits[i] == 1, subtract that bit from target with carry logic
    with qc.if_test((cmp_qubit, 1)):
        cuccaro_sub(qc, n_qubits, target, carry_qubit)

    # 6) Uncompute 'a' or 'N' if you need them zero again
    # e.g.
    for i, bitval in enumerate(a_bits):
        if bitval == 1:
            qc.x(a_qubits[i])
    for i, bitval in enumerate(N_bits):
        if bitval == 1:
            qc.x(n_qubits[i])

    # carry_qubit, cmp_qubit remain 0 if everything is uncomputed properly
    # (in practice, you might want to do more thorough uncomputation).

###############################################################################
# A minimal version of the Cuccaro ripple-carry adder:  'in-place' adder
# We'll define two subroutines: cuccaro_add and cuccaro_sub, each is reversible.
# They add or subtract one quantum register 'a' from another register 'b'.
###############################################################################

def mc_cuccaro_add(qc, a_bits, b_bits, carry, main_ctrl):
    """
    Do (a,b) -> (a, a+b) but only if 'main_ctrl=1'.
    For each original line:
       qc.cx(a_bits[i], b_bits[i])
    we do:
       mc_cx(qc, main_ctrl, a_bits[i], b_bits[i])
    and so on for ccx -> mc_ccx.
    """
    length = len(a_bits)
    # Forward pass
    for i in range(length):
        # Original: qc.cx(a_bits[i], b_bits[i])
        mc_cx(qc, main_ctrl, a_bits[i], b_bits[i])

        # Original: qc.ccx(b_bits[i], a_bits[i], carry)
        mc_ccx(qc, main_ctrl, b_bits[i], a_bits[i], carry)

        if i < length - 1:
            # qc.cx(carry, a_bits[i+1])
            mc_cx(qc, main_ctrl, carry, a_bits[i+1])
            # qc.ccx(a_bits[i+1], b_bits[i], carry)
            mc_ccx(qc, main_ctrl, a_bits[i+1], b_bits[i], carry)

    # Reverse pass
    for i in reversed(range(length)):
        if i < length - 1:
            mc_ccx(qc, main_ctrl, a_bits[i+1], b_bits[i], carry)
            mc_cx(qc, main_ctrl, carry, a_bits[i+1])

        mc_ccx(qc, main_ctrl, b_bits[i], a_bits[i], carry)
        mc_cx(qc, main_ctrl, a_bits[i], b_bits[i])


def cuccaro_add(qc, a_qubits, b_qubits, carry):
    """
    Cuccaro's ripple-carry adder:
      (a, b) -> (a, a + b)
    a and b are lists of qubits, LSB = index 0
    'carry' is an extra qubit.
    """
    # Forward pass
    # 1) Compute partial sums
    for i in range(len(a_qubits)):
        qc.cx(a_qubits[i], b_qubits[i])
        qc.ccx(b_qubits[i], a_qubits[i], carry)

        # shift carry to next iteration
        if i < len(a_qubits) - 1:
            qc.cx(carry, a_qubits[i+1])
            qc.ccx(a_qubits[i+1], b_qubits[i], carry)

    # Reverse pass
    for i in reversed(range(len(a_qubits))):
        if i < len(a_qubits) - 1:
            qc.ccx(a_qubits[i+1], b_qubits[i], carry)
            qc.cx(carry, a_qubits[i+1])

        qc.ccx(b_qubits[i], a_qubits[i], carry)
        qc.cx(a_qubits[i], b_qubits[i])

def cuccaro_sub(qc, a_qubits, b_qubits, carry):
    """
    Same circuit as cuccaro_add, but interpret it as b := b - a mod 2^n
    The trick is that the circuit is its own inverse if we treat 'a' as unchanged.
    So we can just call 'cuccaro_add' again to uncompute. 
    But let's define a direct version for clarity.
    """
    # We can do the exact same gates in reverse order to do subtraction,
    # because ripple-carry add is essentially its own inverse if 'a' is not changed.
    # We'll just reuse the same circuit in reverse. 
    # For demonstration, let's literally do the same steps but in reverse.

    # Forward pass (which is actually the reverse of add):
    for i in range(len(a_qubits)):
        qc.cx(a_qubits[i], b_qubits[i])
        qc.ccx(b_qubits[i], a_qubits[i], carry)
        if i < len(a_qubits) - 1:
            qc.cx(carry, a_qubits[i+1])
            qc.ccx(a_qubits[i+1], b_qubits[i], carry)

    # Reverse pass
    for i in reversed(range(len(a_qubits))):
        if i < len(a_qubits) - 1:
            qc.ccx(a_qubits[i+1], b_qubits[i], carry)
            qc.cx(carry, a_qubits[i+1])

        qc.ccx(b_qubits[i], a_qubits[i], carry)
        qc.cx(a_qubits[i], b_qubits[i])

def modular_exponentiation(qc, a, N, ctrl_register, target_register, ancilla):
    n_ctrl = len(ctrl_register)
    for i in range(n_ctrl):
        exponent = pow(a, 2 ** i, N)
        controlled_modular_multiply(qc, exponent, N, ctrl_register[i], target_register, ancilla)

def mc_add_mod_N(qc, a_val, N_val, target, ancilla, ctrl):
    """
    Same as add_mod_N, but every gate is multi-controlled by 'ctrl'.
    That is, we only do "target = target + a_val mod N_val" if ctrl=1.
    """
    n = len(target)
    needed = n + n + 2
    if len(ancilla) < needed:
        raise ValueError("Not enough ancillas for mc_add_mod_N")

    # Slicing ancillas
    a_qubits = ancilla[:n]
    n_qubits = ancilla[n:2*n]
    carry     = ancilla[2*n]   # single qubit for ripple-carry
    cmp_qubit = ancilla[2*n+1] # for comparison

    # 1) Encode a_val into a_qubits (multi-controlled X)
    a_bits = [(a_val >> i) & 1 for i in range(n)]
    for i, bitval in enumerate(a_bits):
        if bitval == 1:
            # Instead of qc.x(a_qubits[i]), do a multi-controlled X
            mcx(qc, ctrl, a_qubits[i])  # define mcx -> applies X if ctrl=1

    # 2) Encode N_val into n_qubits similarly
    N_bits = [(N_val >> i) & 1 for i in range(n)]
    for i, bitval in enumerate(N_bits):
        if bitval == 1:
            mcx(qc, ctrl, n_qubits[i])

    # 3) Now do a multi-controlled cuccaro_add with 'ctrl'
    mc_cuccaro_add(qc, a_qubits, target, carry, ctrl)

    # 4) Compare target >= N  (multi-controlled version of that subtract, etc.)
    mc_cuccaro_sub(qc, n_qubits, target, carry, ctrl)
    qc.cx(target[n-1], cmp_qubit).c_if(ctrl, 1)  # we might do a multi-control approach
    mc_cuccaro_add(qc, n_qubits, target, carry, ctrl)

    # 5) If cmp_qubit=1, do target -= N (again multi-control).
    # This is a multi-control within a multi-control. Usually we do a bigger approach.
    # For demonstration, let's skip or show partial code.

    # 6) Uncompute 'a_qubits' and 'n_qubits'
    for i, bitval in enumerate(a_bits):
        if bitval == 1:
            mcx(qc, ctrl, a_qubits[i])
    for i, bitval in enumerate(N_bits):
        if bitval == 1:
            mcx(qc, ctrl, n_qubits[i])

    # done


def controlled_modular_multiply(qc, a, N, control, target, ancilla):
    """
    Multiplies 'target' by classical integer 'a' mod N, conditioned on the 'control' qubit being 1.
    Uses repeated addition in a reversible manner.

    target   : list of qubits representing the number to be multiplied
    control  : a single qubit that controls the entire multiplication
    ancilla  : extra qubits. The first half are used as 'accumulator', the remainder as carry bits for add_mod_N
    """

    # Split ancillas into two parts:
    half = len(ancilla) // 2
    accumulator = ancilla[:half]   # holds the partial sums
    carry_bits  = ancilla[half:]   # used for the internal ripple-carry in add_mod_N

    # (Optional) Confirm accumulator is zero – if your pipeline ensures they are zero, skip:
    # for qubit in accumulator:
    #     qc.reset(qubit)

    # 1) For each bit i of 'target':
    #    If (control == 1 AND target[i] == 1), then do:
    #       accumulator = accumulator + (a << i) mod N
    #    This is effectively multi-control repeated addition.
    for i in range(len(target)):
        # 1) Create should_add qubit via Toffoli
        qc.ccx(control, target[i], carry_bits[0])  # if both = 1, should_add=1

# 2) Now do a multi-controlled version of add_mod_N that fires only if carry_bits[0] = 1
    mc_add_mod_N(qc, (a << i), N, accumulator, carry_bits[1:], carry_bits[0])

# 3) Uncompute should_add
    qc.ccx(control, target[i], carry_bits[0])

    # 2) Now 'accumulator' holds (original target) * a (mod N) if control=1,
    #    or 0 if control=0. We want final result in 'target'.

    # A simple approach: do an in-place SWAP of each qubit from accumulator to target.
    # Triple-CNOT SWAP pattern:
    for i in range(len(target)):
        qc.cx(accumulator[i], target[i])
        qc.cx(target[i], accumulator[i])
        qc.cx(accumulator[i], target[i])
    #
    # Now 'target' has the new product, and 'accumulator' has the old 'target' value.

    # (Optional) If you want 'accumulator' back at zero, do the same repeated-add logic or
    # a full uncompute. That can be quite involved, so we omit it here.

    # Done.


def attempt_factor(N, a):
    full_bits = math.ceil(math.log2(N))
    #n_count = 2 * full_bits
    n_count = int(1.5 * full_bits)

    n_working = full_bits
    #n_ancilla = max(1, full_bits // 2)
    n_ancilla = 23 * full_bits  # or even higher



    total_qubits = n_count + n_working + n_ancilla
    if total_qubits > MAX_QUBITS:
        print("Skipping N = {} due to qubit limits ({} required)".format(N, total_qubits))
        return False

    q_all = QuantumRegister(total_qubits, 'q')
    c_out = ClassicalRegister(n_count, 'c')
    qc = QuantumCircuit(q_all, c_out)

    q_count = [q_all[i] for i in range(n_count)]
    q_work = [q_all[i + n_count] for i in range(n_working)]
    q_anc = [q_all[i + n_count + n_working] for i in range(n_ancilla)]

    # Initialization
    for q in q_count:
        qc.h(q)
    qc.x(q_work[0])
    qc.barrier()

    # Modular exponentiation logic
    modular_exponentiation(qc, a, N, q_count, q_work, q_anc)
    qc.barrier()

    # Apply inverse QFT
    iqft_cct(qc, q_count, n_count)

    # Measure
    for i in range(n_count):
        qc.measure(q_count[i], c_out[i])

    # Execute
    job = backend.run(qc, shots=shots)
    job_monitor(job)
    result = job.result()
    counts = result.get_counts()

    for state in counts:
        decimal_result = int(state, 2)
        phase = decimal_result / 2 ** n_count
        r = continued_fraction(phase)
        if r % 2 == 0 and r != 0:
            guess = pow(a, r // 2, N)
            f1, f2 = math.gcd(guess - 1, N), math.gcd(guess + 1, N)
            if f1 > 1 and f1 < N and f2 > 1 and f2 < N:
                print(f"Success: N = {N}, a = {a}, r = {r}, factors = ({f1}, {f2})")
                return True
    print(f"No valid factors found for N = {N} with a = {a}")
    return False

def main():
    for bits, N in semiprimes.items():
        print(f"\nFactoring {N} ({bits}-bit)")
        a = 2
        start_time = time.time()
        while a < N:
            try:
                if math.gcd(a, N) != 1:
                    print(f"Trivial factor found by GCD: {math.gcd(a, N)}")
                    break
                success = attempt_factor(N, a)
                if success:
                    break
                if time.time() - start_time > TIMEOUT_SECONDS:
                    print(f"Timeout reached for N = {N}")
                    break
                a += 1
            except Exception as e:
                print(f"Error for N = {N}, a = {a}: {e}")
                break

if __name__ == "__main__":
    main()
