import time
import requests
import matplotlib.pyplot as plt
import statistics
import psutil

SHORT_TEXT = 'елочка'
LONG_TEXT = (
    'В лесу родилась елочка, в лесу она росла.'
    'Зимой и летом стройная, зеленая была.'
    'Метель ей пела песенку: Спи, елочка, бай-бай!'
    'Мороз снежком укутывал: Смотри, не замерзай!'
)


def request_sender(short=True, url='http://localhost:5000', id_num=None):
    """Sending request to server
    Args:
        short (bool, optional): True if short text, False if long text
        url (str, optional): Url for sending request
        id_num: int or None: Current request id
    Returns:
          dict: dictionary with keys:
            latency (ms): full request time
            id: current request id
            inference_time: model work time, in ms (0in case of error)
            code: 1 - success request, 0 - otherwise
    """
    if short:
        text = SHORT_TEXT
    else:
        text = LONG_TEXT
    start = time.perf_counter()
    try:
        response = requests.post(
            url + '/embed',
            json={'text': text},
            timeout=10
        )
        end = time.perf_counter()
        latency_ms = (end - start) * 1000
        data = response.json()
        return {
            'latency': latency_ms,
            'id': id_num,
            'inference_time': data.get('inference_time', 0),
            'code': 1,
        }
    except Exception as e:
        end = time.perf_counter()
        latency_ms = (end - start) * 1000
        return {
            'latency': latency_ms,
            'id': id_num,
            'inference_time': 0,
            'code': 0,
        }


def percentile(data, p):
    """
    Calculate percentile p in data
    Args:
        data (array): data for percentile calculating
        p (int): needed percentile
    Returns:
        float: p-percentile of data
    """
    data_sorted = sorted(data)
    n = len(data_sorted)
    if n == 0:
        return 0
    index = (p / 100) * (n - 1)
    lower = int(index)
    if lower == n - 1:
        return data_sorted[-1]
    upper = lower + 1
    weight = index - lower
    return data_sorted[lower] * (1 - weight) + data_sorted[upper] * weight


def cpu_ram_test(count=100, short=True):
    """
    Make <count> requests to server for CPU and RAM benchmark
    Args:
         count (int): total count of requests
         short (bool, optional): True if short text, False if long text
    Returns:
        cpu (list): list with cpu usage
        ram (list): list with ram usage
    """
    cpu = []
    ram = []
    for id_num in range(count):
        cpu_begin = psutil.cpu_percent(interval=0.1)
        ram_begin = psutil.virtual_memory().percent
        request_sender(short=short, id_num=id_num)
        cpu_end = psutil.cpu_percent(interval=0.1)
        ram_end = psutil.virtual_memory().percent
        cpu_mean = (cpu_end + cpu_begin) / 2
        ram_mean = (ram_end + ram_begin) / 2
        cpu.append(cpu_mean)
        ram.append(ram_mean)
    return cpu, ram


def test_load(count=100, short=True):
    """
    Make <count> requests to server
    Args:
         count (int): total count of requests
         short (bool, optional): True if short text, False if long text
    Returns:
        result (list of dicts): list with server answers
    """
    result = []
    for id_num in range(count):
        result.append(request_sender(short=short, id_num=id_num))
    return result


def warm_up():
    """Function for warm up
    Args: None
    Returns: None
    """
    for _ in range(5):
        request_sender(short=True)
        request_sender(short=False)


def save_txt_report(
        latency_short, latency_long,
        inference_time_short, inference_time_long,
        cpu_short, ram_short, cpu_long, ram_long,
        success_rate_short, success_rate_long,
        filename='report.txt'
):
    """
    Сохраняет отчёт с метриками в  txt файл
    """
    p50_short = percentile(latency_short, 50)
    p95_short = percentile(latency_short, 95)
    p99_short = percentile(latency_short, 99)
    p50_long = percentile(latency_long, 50)
    p95_long = percentile(latency_long, 95)
    p99_long = percentile(latency_long, 99)

    total_time_short_sec = sum(latency_short) / 1000
    total_time_long_sec = sum(latency_long) / 1000
    rps_short = len(latency_short) / total_time_short_sec
    rps_long = len(latency_long) / total_time_long_sec

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("BENCHMARK REPORT\n")
        # SHORT TEXT
        f.write("=" * 70 + "\n")
        f.write(" SHORT TEXT RESULTS\n")
        f.write(
            f'Percent of success requests: {100 * success_rate_short:.1f}\n')
        f.write(" LATENCY (ms):\n")
        f.write(f"Mean:{statistics.mean(latency_short):8.2f} ms\n")
        f.write(f"P50: {p50_short:8.2f} ms\n")
        f.write(f"P95: {p95_short:8.2f} ms\n")
        f.write(f"P99: {p99_short:8.2f} ms\n")
        f.write(f"Min: {min(latency_short):8.2f} ms\n")
        f.write(f"Max: {max(latency_short):8.2f} ms\n")
        if len(latency_short) > 1:
            f.write(f" Std:  {statistics.stdev(latency_short):8.2f} ms\n")

        f.write("\n INFERENCE TIME (ms):\n")
        if inference_time_short and any(inference_time_short):
            f.write(
                f"Mean: {statistics.mean(inference_time_short):8.2f} ms\n"
            )
            f.write(
                f"Min:  {min(inference_time_short):8.2f} ms\n"
            )
            f.write(
                f"Max:  {max(inference_time_short):8.2f} ms\n"
            )
        else:
            f.write("No inference time data available\n")

        f.write("\n CPU USAGE(%):\n")
        f.write(f"Mean: {statistics.mean(cpu_short):8.1f} %\n")
        f.write(f"Min:  {min(cpu_short):8.1f} %\n")
        f.write(f"Max:  {max(cpu_short):8.1f} %\n")

        f.write("\n RAM USAGE(%):\n")
        f.write(f"Mean: {statistics.mean(ram_short):8.1f} %\n")
        f.write(f"Min:  {min(ram_short):8.1f} %\n")
        f.write(f"Max:  {max(ram_short):8.1f} %\n")

        f.write("\n RPS:\n")
        f.write(f"RPS:  {rps_short:8.2f} requests/second\n")

        # LONG TEXT
        f.write("\n" + "=" * 70 + "\n")
        f.write(" LONG TEXT RESULTS\n")
        f.write(
            f'Percent of success requests: {100 * success_rate_long:.1f}\n')
        f.write(" LATENCY (ms):\n")
        f.write(f"Mean:  {statistics.mean(latency_long):8.2f} ms\n")
        f.write(f"P50:   {p50_long:8.2f} ms\n")
        f.write(f"P95:   {p95_long:8.2f} ms\n")
        f.write(f"P99:   {p99_long:8.2f} ms\n")
        f.write(f"Min:   {min(latency_long):8.2f} ms\n")
        f.write(f"Max:   {max(latency_long):8.2f} ms\n")
        if len(latency_long) > 1:
            f.write(f"Std:   {statistics.stdev(latency_long):8.2f} ms\n")

        f.write("\n INFERENCE TIME (ms):\n")
        if inference_time_long and any(inference_time_long):
            f.write(
                f"Mean:  {statistics.mean(inference_time_long):8.2f} ms\n"
            )
            f.write(
                f"Min:   {min(inference_time_long):8.2f} ms\n"
            )
            f.write(
                f"Max:   {max(inference_time_long):8.2f} ms\n"
            )
        else:
            f.write("  ️  No inference time data available\n")

        f.write("\n CPU USAGE(%):\n")
        f.write(f"Mean:   {statistics.mean(cpu_long):8.1f} %\n")
        f.write(f"Min:      {min(cpu_long):8.1f} %\n")
        f.write(f"Max:      {max(cpu_long):8.1f} %\n")

        f.write("\n RAM USAGE(%):\n")
        f.write(f"Mean:     {statistics.mean(ram_long):8.1f} %\n")
        f.write(f"Min:      {min(ram_long):8.1f} %\n")
        f.write(f"Max:      {max(ram_long):8.1f} %\n")

        f.write("\n RPS:\n")
        f.write(f"RPS:      {rps_long:8.2f} requests/second\n")


if __name__ == "__main__":
    print('CPU & RAM tests started')
    warm_up()
    cpu_test_count = 300
    cpu_short, ram_short = cpu_ram_test(count=cpu_test_count, short=True)
    cpu_long, ram_long = cpu_ram_test(count=cpu_test_count, short=False)

    print('Load_tests started')
    warm_up()
    test_load_count = 300
    short_text_res = test_load(count=test_load_count, short=True)
    long_text_res = test_load(count=test_load_count, short=False)

    inference_time_short = [
        item['inference_time'] for item in short_text_res if item['code'] == 1]
    inference_time_long = [
        item['inference_time'] for item in long_text_res if item['code'] == 1]

    latency_short = [
        item['latency'] for item in short_text_res if item['code'] == 1]
    latency_long = [
        item['latency'] for item in long_text_res if item['code'] == 1]

    success_rate_short = (len(latency_short)) / test_load_count
    success_rate_long = (len(latency_long)) / test_load_count

    if len(latency_short) == 0 or len(latency_long) == 0:
        print("All request answers are not succesfull")
        exit(1)

    print('Graphs generating')
    # Построение графиков
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(
        3, 2, figsize=(12, 8))

    # 1. Current latency
    ax1.plot(latency_short, label='short text', color='red', alpha=0.7)
    ax1.plot(latency_long, label='long text', color='green', alpha=0.7)
    mean_latency_short = statistics.mean(latency_short)
    mean_latency_long = statistics.mean(latency_long)
    ax1.axhline(
        mean_latency_short,
        color='r',
        linestyle='--',
        label='mean_short_text'
    )
    ax1.axhline(
        mean_latency_long,
        color='g',
        linestyle='--',
        label='mean_long_text'
    )
    ax1.text(
        len(latency_short),
        mean_latency_short - 1,
        f'{mean_latency_short:.1f}'
    )
    ax1.text(
        len(latency_long),
        mean_latency_long + 1,
        f'{mean_latency_long:.1f}'
    )
    ax1.grid()
    ax1.legend()
    ax1.set_title('Latency')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_xlabel('request_id')

    # 2. Latency with percentiles
    ax2.plot(sorted(latency_short), label='short text', color='red', alpha=0.7)
    ax2.plot(sorted(latency_long), label='long text', color='g', alpha=0.7)

    n_long = len(latency_long)

    pos_50 = int(n_long * 0.5)
    pos_95 = int(n_long * 0.95)
    pos_99 = int(n_long * 0.99)

    ax2.axvline(x=pos_50, color='black', linestyle=':', label='perc 50th')
    ax2.axvline(x=pos_95, color='black', linestyle='-.', label='perc 95th')
    ax2.axvline(x=pos_99, color='black', linestyle='--', label='perc 99th')

    ax2.grid()
    ax2.legend()
    ax2.set_xlabel('requests')
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('Latency percentiles')

    # 3. Histograms latency
    bins = list(range(0, 201, 10))  # 0, 10, 20, ..., 200

    ax3.hist(latency_short, bins=bins, color='r', alpha=0.5, label='shor_texе')
    ax3.hist(latency_long, bins=bins, color='g', alpha=0.5, label='long_text')
    ax3.set_title('Latency histograms')
    ax3.set_xlabel('Latency')
    ax3.set_ylabel('Count')
    ax3.grid()
    ax3.legend()

    # 4: Total Latency & model inference time (boxplot)
    mean_total_short = statistics.mean(latency_short)
    mean_total_long = statistics.mean(latency_long)
    mean_inf_short = (
        statistics.mean(inference_time_short)
        if inference_time_short
        else 0
    )
    mean_inf_long = (
        statistics.mean(inference_time_long)
        if inference_time_long
        else 0
    )

    data_to_plot = [
        latency_short,
        inference_time_short,
        latency_long,
        inference_time_long
    ]
    positions = [1, 2, 3, 4]
    colors = ['red', 'salmon', 'green', 'lightgreen']
    labels = ['Total\nshort', 'Model\nshort', 'Total\nlong', 'Model\nlong']

    bp = ax4.boxplot(data_to_plot,
                     positions=positions,
                     widths=0.6,
                     patch_artist=True,
                     showfliers=False
                     )

    ax4.set_xticks(positions)
    ax4.set_xticklabels(labels)
    ax4.set_ylabel('Time (ms)')
    ax4.set_title('Total Latency and Model inference time')
    ax4.grid(True, alpha=0.3, axis='y')

    ax4.text(
        1, mean_latency_short + 2, f'{mean_latency_short:.1f}', fontsize=8)
    ax4.text(
        2, mean_inf_short + 2, f'{mean_inf_short:.1f}', fontsize=8)
    ax4.text(
        3, mean_latency_long + 2, f'{mean_latency_long:.1f}', fontsize=8)
    ax4.text(
        4, mean_inf_long + 2, f'{mean_inf_long:.1f}', fontsize=8)

    # График 5: CPU Usage
    ax5.plot(cpu_short, color='red', alpha=0.7, label='short text')
    ax5.plot(cpu_long, color='green', alpha=0.7, label='long text')
    ax5.axhline(y=statistics.mean(cpu_short), c='r', linestyle='--', alpha=0.5)
    ax5.axhline(y=statistics.mean(cpu_long), c='g', linestyle='--', alpha=0.5)
    ax5.set_title('CPU Usage')
    ax5.set_xlabel('Measurement points')
    ax5.set_ylabel('CPU (%)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # График 6: RAM Usage
    ax6.plot(ram_short, color='red', alpha=0.7, label='short text')
    ax6.plot(ram_long, color='green', alpha=0.7, label='long text')
    ax6.axhline(y=statistics.mean(ram_short), c='r', linestyle='--', alpha=0.5)
    ax6.axhline(y=statistics.mean(ram_long), c='g', linestyle='--', alpha=0.5)
    ax6.set_title('RAM usage')
    ax6.set_xlabel('Measurement points')
    ax6.set_ylabel('RAM(%)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    print('Text report saving')
    save_txt_report(
        latency_short, latency_long,
        inference_time_short, inference_time_long,
        cpu_short, ram_short, cpu_long, ram_long,
        success_rate_short, success_rate_long,
        filename='report.txt'
    )

    plt.tight_layout()
    plt.savefig('benchmark.png')
    plt.show()
