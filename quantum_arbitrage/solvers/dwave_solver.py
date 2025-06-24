import numpy as np
import pandas as pd
from dwave.samplers import SimulatedAnnealingSampler
from tqdm import tqdm

from quantum_arbitrage.solvers.dwaveutils.dwaveutils import bl_lstsq
from quantum_arbitrage.solvers.dwaveutils.dwaveutils.utils import Binary2Float


def solve_qubo(Dimension, qubits, A, b, true_x, scale):
    A = [[col * scale for col in row] for row in A]
    b = [el * scale for el in b]
    true_x = [el * scale for el in true_x]

    QM = np.zeros(((qubits + 1) * Dimension, (qubits + 1) * Dimension))
    ### Linear terms ###
    for k in range(Dimension):
        for i in range(Dimension):
            cef1 = pow(2, 2 * qubits) * pow(A[k][i], 2)
            cef2 = pow(2, qubits + 1) * A[k][i] * b[k]
            po2 = (qubits + 1) * i + qubits
            QM[po2][po2] = QM[po2][po2] + cef1 + cef2
            for l in range(qubits):
                cef1 = pow(2, 2 * l) * pow(A[k][i], 2)
                cef2 = pow(2, l + 1) * A[k][i] * b[k]
                po1 = (qubits + 1) * i + l
                QM[po1][po1] = QM[po1][po1] + cef1 - cef2

    print('Generating first quadratic term...')
    ### First quadratic term ###
    for k in tqdm(range(Dimension)):
        for i in range(Dimension):
            for l in range(qubits):
                qcef = pow(2, l + qubits + 1) * pow(A[k][i], 2)
                po3 = (qubits + 1) * i + l
                po4 = (qubits + 1) * i + qubits
                QM[po3][po4] = QM[po3][po4] - qcef
            for l1 in range(qubits - 1):
                for l2 in range(l1 + 1, qubits):
                    qcef = pow(2, l1 + l2 + 1) * pow(A[k][i], 2)
                    po1 = (qubits + 1) * i + l1
                    po2 = (qubits + 1) * i + l2
                    QM[po1][po2] = QM[po1][po2] + qcef

    print('\n\nGenerating second quadratic term...')
    ### Second quadratic term ###
    for k in tqdm(range(Dimension)):
        for i in range(Dimension - 1):
            for j in range(i + 1, Dimension):
                qcef = pow(2, 2 * qubits + 1) * A[k][i] * A[k][j]
                po3 = (qubits + 1) * i + qubits
                po4 = (qubits + 1) * j + qubits
                QM[po3][po4] = QM[po3][po4] + qcef
                for l in range(qubits):
                    qcef = pow(2, l + qubits + 1) * A[k][i] * A[k][j]
                    po5 = (qubits + 1) * i + qubits
                    po6 = (qubits + 1) * j + l
                    QM[po5][po6] = QM[po5][po6] - qcef
                    po7 = (qubits + 1) * i + l
                    po8 = (qubits + 1) * j + qubits
                    QM[po7][po8] = QM[po7][po8] - qcef
                for l1 in range(qubits):
                    for l2 in range(qubits):
                        qcef = pow(2, l1 + l2 + 1) * A[k][i] * A[k][j]
                        po1 = (qubits + 1) * i + l1
                        po2 = (qubits + 1) * j + l2
                        QM[po1][po2] = QM[po1][po2] + qcef

    # sampler_auto = EmbeddingComposite(DWaveSampler(solver={'qpu': True}))
    sampler_auto = SimulatedAnnealingSampler()

    Q = {}
    epsilon = 1e-10  # tiny bias to force inclusion
    for i in range(QM.shape[0]):
        # add linear term, if zero add epsilon
        val = QM[i][i]
        if val == 0:
            val = epsilon
        Q[(f'q{i}', f'q{i}')] = val
        for j in range(i + 1, QM.shape[1]):
            if QM[i][j] != 0:
                Q[(f'q{i}', f'q{j}')] = QM[i][j]

    print('\n\nSolving QUBO with Simulated annealing sampler...')
    # number of reads for Simulated annealing (SA) or Quantum annealing (QA)
    num_reads = 1000
    sampleset = sampler_auto.sample_qubo(Q, num_reads=num_reads)

    # convert sampleset and its aggregate version to dataframe
    sampleset_pd = sampleset.to_pandas_dataframe()
    sampleset_pd_agg = sampleset.aggregate().to_pandas_dataframe()
    num_states = len(sampleset_pd_agg)
    num_x_entry = Dimension
    qubit_cols = [col for col in sampleset_pd.columns if col.startswith('q')]
    num_q_entry = len(qubit_cols)

    # set the bit value to discrete the actual value as a fixed point
    num_bits = num_q_entry // Dimension
    bit_value = bl_lstsq.get_bit_value(num_bits, fixed_point=True, sign="p")
    # discretized version of matrix `A`
    A_discrete = bl_lstsq.discretize_matrix(A, bit_value)

    # concatenate `sampleset_pd_agg` and `x_at_each_state`
    x_at_each_state = pd.DataFrame(
        np.row_stack(
            [(sampleset_pd_agg.iloc[i][:num_q_entry]).values.reshape(
                (num_x_entry, -1)) @ bit_value
             for i in range(num_states)]
        ),
        columns=['x' + str(i) for i in range(num_x_entry)]
    )
    sampleset_pd_agg = pd.concat([sampleset_pd_agg, x_at_each_state], axis=1)
    sampleset_pd_agg.rename(
        columns=lambda c: c if isinstance(c, str) else 'q' + str(c),
        inplace=True
    )
    # lowest energy state x and q
    lowest_q = sampleset_pd_agg.sort_values(
        'energy').iloc[0, :num_q_entry].values
    lowest_x = Binary2Float.to_fixed_point(lowest_q, bit_value)
    # frequently occurring x and q
    frequent_q = sampleset_pd_agg.sort_values(
        'num_occurrences', ascending=False).iloc[0, :num_q_entry].values
    frequent_x = Binary2Float.to_fixed_point(frequent_q, bit_value)
    # calculate expected x from x
    expected_x = sampleset_pd_agg.apply(
        lambda row: row.iloc[-num_x_entry:]
                    * (row.num_occurrences / num_reads),
        axis=1
    ).sum().values
    # calculate excepted x from q
    tmp_q = sampleset_pd_agg.apply(
        lambda row: row.iloc[:num_q_entry]
                    * (row.num_occurrences / num_reads),
        axis=1
    ).sum() > 0.5  # bool
    expected_x_discrete = Binary2Float.to_fixed_point(np.array(tmp_q), bit_value)

    true_b = b
    print('=' * 50)
    print('true A:', [[float(x) for x in row] for row in A])
    print('true x:', [float(x) for x in true_x])
    print('true b:', [float(x) for x in true_b])
    print('bit value:', bit_value)

    print('=' * 50)
    print('# Simulated annealing/Quantum annealing')
    print('lowest energy state x:')
    print(lowest_x)
    print('lowest energy state q:')
    print(lowest_q)
    print('b:', A @ lowest_x)
    print('2-norm:', np.linalg.norm(A @ lowest_x - true_b))
    print('-' * 50)
    print('most frequently occurring x:')
    print(frequent_x)
    print('most frequently occurring q:')
    print(frequent_q)
    print('b:', A @ frequent_x)
    print('2-norm:', np.linalg.norm(A @ frequent_x - true_b))
    print('-' * 50)
    print('expected x (from real value):')
    print(expected_x)
    print('b:', A @ expected_x)
    print('2-norm:', np.linalg.norm(A @ expected_x - true_b))
    print('-' * 50)
    print('expected x (from discrete value):')
    print(expected_x_discrete)
    print('b:', A @ expected_x_discrete)
    print('2-norm:', np.linalg.norm(A @ expected_x_discrete - true_b))
    print('-' * 50)
    print('Sample set:')
    print(sampleset_pd_agg.sort_values('num_occurrences', ascending=False))
    print('=' * 50)

    return [x / scale for x in lowest_x]
