// --- Global Constants & DOM Elements ---
const EPSILON = 1e-6;

const marketPricesInput = document.getElementById('market-prices');
const trueProbsInput = document.getElementById('true-probs');
const initialWealthInput = document.getElementById('initial-wealth');
const learningRateInput = document.getElementById('learning-rate');
const iterationsInput = document.getElementById('iterations');
const batchSizeInput = document.getElementById('batch-size');
const runButton = document.getElementById('run-optimization');
const logConsole = document.getElementById('log-console');
const resultsSummaryDiv = document.getElementById('results-summary');
const resultsTbody = document.getElementById('results-tbody');

// --- Helper Functions ---

function log(message) {
    logConsole.textContent += message + '\n';
    logConsole.scrollTop = logConsole.scrollHeight;
}

function clearLog() {
    logConsole.textContent = '';
}

function parseCsv(input) {
    return input.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
}

function formatNum(num, digits = 4) {
    if (isNaN(num)) return 'NaN';
    return num.toFixed(digits);
}

// --- Main Optimization Logic ---

async function runOptimization() {
    clearLog();
    log('Starting optimization...');
    runButton.disabled = true;
    runButton.textContent = 'Optimizing...';

    // 1. Get and Validate Inputs
    const P_values = parseCsv(marketPricesInput.value);
    const Y_prob_values = parseCsv(trueProbsInput.value);
    const W = parseFloat(initialWealthInput.value);
    const learningRate = parseFloat(learningRateInput.value);
    const iterations = parseInt(iterationsInput.value);
    const batchSize = parseInt(batchSizeInput.value);

    if (P_values.length === 0 || Y_prob_values.length === 0 || P_values.length !== Y_prob_values.length) {
        log('Error: Market Prices and True Probabilities must have the same number of comma-separated values.');
        runButton.disabled = false;
        runButton.textContent = 'Run Optimization';
        return;
    }
    const numEvents = P_values.length;
    log(`Model parameters: ${numEvents} events, W=$${W}, LR=${learningRate}, Iterations=${iterations}, Batch Size=${batchSize}`);

    // 2. Initialize Tensors and Variables
    const P = tf.tensor1d(P_values);
    const P_adj = P.clipByValue(EPSILON, 1 - EPSILON);
    const Y_prob = tf.tensor1d(Y_prob_values);

    // Optimization parameters (unconstrained)
    const theta_yes = tf.variable(tf.scalar(0.0));
    const theta_no = tf.variable(tf.scalar(0.0));
    const phi_yes = tf.variable(tf.zeros([numEvents]));
    const phi_no = tf.variable(tf.zeros([numEvents]));
    const optimizer = tf.train.adam(learningRate);

    // 3. Define the Loss Function (Negative Expected Log Return)
    const lossFunction = () => {
        return tf.tidy(() => {
            const Y_outcome_batch = tf.randomUniform([batchSize, numEvents]).less(Y_prob).asType('float32');
            const thetas = tf.stack([theta_yes, theta_no]).squeeze();
            const f_fractions = tf.softmax(thetas);
            const f_yes = f_fractions.slice(0, 1);
            const f_no = f_fractions.slice(1, 1);
            const alpha_yes = tf.softmax(phi_yes);
            const alpha_no = tf.softmax(phi_no);
            const n_yes = alpha_yes.mul(f_yes).mul(W).div(P_adj);
            const n_no = alpha_no.mul(f_no).mul(W).div(tf.sub(1, P_adj));
            const profit_per_event = Y_outcome_batch.sub(P).mul(n_yes.sub(n_no));
            const G_batch = profit_per_event.sum(1);
            const wealth_ratio_batch = G_batch.div(W).add(1);
            const log_return_batch = tf.log(wealth_ratio_batch.clipByValue(EPSILON, Infinity));
            const expected_log_return = log_return_batch.mean();
            return expected_log_return.neg();
        });
    };

    // 4. Optimization Loop (async)
    try {
        for (let i = 0; i < iterations; i++) {
            // *** THIS IS THE CORRECTED LINE ***
            // optimizer.minimize returns the loss tensor directly, not an object.
            const loss = optimizer.minimize(lossFunction, true, [theta_yes, theta_no, phi_yes, phi_no]);
            
            const lossValue = await loss.data();
            loss.dispose();

            const expectedLogReturn = -lossValue[0];

            if (isNaN(expectedLogReturn)) {
                log(`\n--- ERROR ---`);
                log(`Optimization failed at iteration ${i}: Result is NaN (Not a Number).`);
                log(`This is likely due to an overly large learning rate or unstable input values.`);
                log(`Try lowering the Learning Rate or simplifying the market inputs.`);
                log(`--- STOPPING OPTIMIZATION ---`);
                return;
            }

            if (i < 10 || i % 100 === 0 || i === iterations - 1) {
                log(`Iter ${i}: Expected Log Return = ${formatNum(expectedLogReturn, 6)}`);
                await tf.nextFrame();
            }
        }
        log('Optimization finished successfully.');
        await displayResults({ P, P_adj, W, theta_yes, theta_no, phi_yes, phi_no });
    } catch (error) {
        log(`An unexpected error occurred: ${error.message}`);
        console.error(error);
    } finally {
        runButton.disabled = false;
        runButton.textContent = 'Run Optimization';
        
        P.dispose();
        P_adj.dispose();
        Y_prob.dispose();
        theta_yes.dispose();
        theta_no.dispose();
        phi_yes.dispose();
        phi_no.dispose();
    }
}


// --- Display Results ---
async function displayResults(params) {
    const { P, P_adj, W, theta_yes, theta_no, phi_yes, phi_no } = params;

    const { f_yes, f_no, alpha_yes, alpha_no, n_yes_net, n_no_net, cash_from_cancel } = tf.tidy(() => {
        const thetas = tf.stack([theta_yes, theta_no]).squeeze();
        const f_fractions = tf.softmax(thetas);
        const f_yes_val = f_fractions.slice(0, 1);
        const f_no_val = f_fractions.slice(1, 1);
        const alpha_yes_val = tf.softmax(phi_yes);
        const alpha_no_val = tf.softmax(phi_no);
        const n_yes = alpha_yes_val.mul(f_yes_val).mul(W).div(P_adj);
        const n_no = alpha_no_val.mul(f_no_val).mul(W).div(tf.sub(1, P_adj));
        const canceled_shares = tf.minimum(n_yes, n_no);
        const n_yes_net_val = n_yes.sub(canceled_shares);
        const n_no_net_val = n_no.sub(canceled_shares);
        const cash_from_cancel_val = canceled_shares.sum();
        return {
            f_yes: f_yes_val.arraySync()[0], f_no: f_no_val.arraySync()[0],
            alpha_yes: alpha_yes_val.arraySync(), alpha_no: alpha_no_val.arraySync(),
            n_yes_net: n_yes_net_val.arraySync(), n_no_net: n_no_net_val.arraySync(),
            cash_from_cancel: cash_from_cancel_val.arraySync()
        };
    });
    
    const p_adj_arr = P_adj.arraySync();
    const total_yes_cost = n_yes_net.reduce((sum, val, i) => sum + val * p_adj_arr[i], 0);
    const total_no_cost = n_no_net.reduce((sum, val, i) => sum + val * (1 - p_adj_arr[i]), 0);
    const total_capital_used_net = total_yes_cost + total_no_cost;
    const cash_unallocated = W - total_capital_used_net - cash_from_cancel;

    resultsSummaryDiv.innerHTML = `
        <p><strong>Fraction to 'Yes' (f_yes):</strong> ${formatNum(f_yes)}</p>
        <p><strong>Fraction to 'No' (f_no):</strong> ${formatNum(f_no)}</p>
        <p><strong>Net Capital Deployed:</strong> $${formatNum(total_capital_used_net, 2)}</p>
        <p><strong>Cash from Canceled Pairs:</strong> $${formatNum(cash_from_cancel, 2)}</p>
        <p><strong>Remaining Unallocated Cash:</strong> $${formatNum(cash_unallocated, 2)}</p>
    `;

    resultsTbody.innerHTML = '';
    const numEvents = alpha_yes.length;
    for (let i = 0; i < numEvents; i++) {
        let netSharesText, capital_total;
        if (n_yes_net[i] > n_no_net[i]) {
            netSharesText = `${formatNum(n_yes_net[i])} Yes`;
            capital_total = n_yes_net[i] * p_adj_arr[i];
        } else if (n_no_net[i] > 0) {
            netSharesText = `${formatNum(n_no_net[i])} No`;
            capital_total = n_no_net[i] * (1 - p_adj_arr[i]);
        } else {
            netSharesText = '0.0000';
            capital_total = 0;
        }
        const row = `
            <tr>
                <td>${i + 1}</td>
                <td>${netSharesText}</td>
                <td>$${formatNum(capital_total, 2)}</td>
                <td>${formatNum(alpha_yes[i])}</td>
                <td>${formatNum(alpha_no[i])}</td>
            </tr>
        `;
        resultsTbody.innerHTML += row;
    }
}

// --- Event Listeners ---
runButton.addEventListener('click', runOptimization);

// Initial state
log('Ready. Enter market data and click "Run Optimization".');
resultsSummaryDiv.innerHTML = '<p>Run optimization to see results.</p>';