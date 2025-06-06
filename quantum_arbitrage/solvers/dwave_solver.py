import numpy as np
import pandas as pd
from dwave.samplers import SimulatedAnnealingSampler

from quantum_arbitrage.solvers.dwaveutils.dwaveutils import bl_lstsq
from quantum_arbitrage.solvers.dwaveutils.dwaveutils.utils import Binary2Float


def solve_qubo(Dimension, qubits, A, b):
    # 6 qubits with integer solutions
    # x1 = q1 + 2q2 - 4q3
    # x = {{-1}, {2}}
    # mininum -26

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

    ### First quadratic term ###
    for k in range(Dimension):
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

    ### Second quadratic term ###
    for k in range(Dimension):
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

    # Print Matrix Q
    print("# Matrix Q is")
    print(QM)
    print("\nMinimum energy is ", -np.dot(b, b))
    print("\n")

    # Print Python code for the run in D-Wave quantum processing unit
    print("Running code for D-Wave:\n")
    print("from dwave.system import DWaveSampler, EmbeddingComposite")
    print("sampler_auto = EmbeddingComposite(DWaveSampler(solver={'qpu': True}))\n")
    print("linear = {", end="")
    for i in range((qubits + 1) * Dimension - 1):
        linear = i + 1
        print("('q", linear, "','q", linear, "'):", format(QM[i][i]), sep='', end=", ")
    print("('q", (qubits + 1) * Dimension, "','q", (qubits + 1) * Dimension, "'):",
          format(QM[(qubits + 1) * Dimension - 1][(qubits + 1) * Dimension - 1]), "}", sep='')

    print("\nquadratic = {", end="")
    for i in range((qubits + 1) * Dimension - 1):
        for j in range(i + 1, (qubits + 1) * Dimension):
            qdrt1 = i + 1
            qdrt2 = j + 1
            if i == (qubits + 1) * Dimension - 2 and j == (qubits + 1) * Dimension - 1:
                print("('q", qdrt1, "','q", qdrt2, "'):", format(QM[i][j]), "}", sep='')
            else:
                print("('q", qdrt1, "','q", qdrt2, "'):", format(QM[i][j]), sep='', end=", ")

    print("\nQ = dict(linear)")
    print("Q.update(quadratic)\n")

    qa_iter = 1000
    print("sampleset = sampler_auto.sample_qubo(Q, num_reads=", qa_iter, ")", sep="")
    print("print(sampleset)")

    # sampler_auto = EmbeddingComposite(DWaveSampler(solver={'qpu': True}))
    sampler_auto = SimulatedAnnealingSampler()

    linear = {('q1', 'q1'): 26.0, ('q2', 'q2'): 72.0, ('q3', 'q3'): 96.0, ('q4', 'q4'): -13.0, ('q5', 'q5'): -16.0,
              ('q6', 'q6'): 152.0}

    quadratic = {('q1', 'q2'): 40.0, ('q1', 'q3'): -80.0, ('q1', 'q4'): 2.0, ('q1', 'q5'): 4.0, ('q1', 'q6'): -8.0,
                 ('q2', 'q3'): -160.0, ('q2', 'q4'): 4.0, ('q2', 'q5'): 8.0, ('q2', 'q6'): -16.0, ('q3', 'q4'): -8.0,
                 ('q3', 'q5'): -16.0, ('q3', 'q6'): 32.0, ('q4', 'q5'): 20.0, ('q4', 'q6'): -40.0, ('q5', 'q6'): -80.0}

    Q = dict(linear)
    Q.update(quadratic)

    sampleset = sampler_auto.sample_qubo(Q, num_reads=1000)
    print(sampleset)

    # set the bit value to discrete the actual value as a fixed point
    num_bits = 4
    bit_value = bl_lstsq.get_bit_value(num_bits, fixed_point=True)
    # discretized version of matrix `A`
    A_discrete = bl_lstsq.discretize_matrix(A, bit_value)
    # number of reads for Simulated annealing (SA) or Quantum annealing (QA)
    num_reads = 1000

    # convert sampleset and its aggregate version to dataframe
    sampleset_pd = sampleset.to_pandas_dataframe()
    sampleset_pd_agg = sampleset.aggregate().to_pandas_dataframe()
    num_states = len(sampleset_pd_agg)
    num_x_entry = Dimension
    num_q_entry = A_discrete.shape[1]
    # concatenate `sampleset_pd` and `x_at_each_read`
    x_at_each_read = pd.DataFrame(
        np.row_stack(
            [(sampleset_pd.iloc[i][:num_q_entry]).values.reshape(
                (num_x_entry, -1)) @ bit_value
             for i in range(num_reads)]
        ),
        columns=['x' + str(i) for i in range(num_x_entry)]
    )
    sampleset_pd = pd.concat([sampleset_pd, x_at_each_read], axis=1)
    sampleset_pd.rename(
        columns=lambda c: c if isinstance(c, str) else 'q' + str(c),
        inplace=True
    )
    # concatnate `sampleset_pd_agg` and `x_at_each_state`
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

    return sampleset
