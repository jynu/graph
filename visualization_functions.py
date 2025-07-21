import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import numpy as np

def create_accuracy_visualizations(results: List[Dict], timestamp: str):
    """Create comprehensive accuracy visualization plots."""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Overall Accuracy Comparison (Top plot)
    ax1 = plt.subplot(3, 3, (1, 3))
    plot_overall_accuracy_comparison(results, ax1)
    
    # 2. Top-K Accuracy Analysis
    ax2 = plt.subplot(3, 3, 4)
    plot_top_k_accuracy(results, ax2)
    
    # 3. Precision vs Recall
    ax3 = plt.subplot(3, 3, 5)
    plot_precision_recall(results, ax3)
    
    # 4. Performance vs Accuracy Trade-off
    ax4 = plt.subplot(3, 3, 6)
    plot_performance_accuracy_tradeoff(results, ax4)
    
    # 5. Confidence Distribution
    ax5 = plt.subplot(3, 3, 7)
    plot_confidence_distribution(results, ax5)
    
    # 6. Method Comparison Heatmap
    ax6 = plt.subplot(3, 3, 8)
    plot_method_comparison_heatmap(results, ax6)
    
    # 7. Query Difficulty Analysis
    ax7 = plt.subplot(3, 3, 9)
    plot_query_difficulty_analysis(results, ax7)
    
    plt.tight_layout(pad=3.0)
    
    # Save the comprehensive plot
    filename = f"accuracy_analysis_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Comprehensive accuracy analysis saved to {filename}")
    
    # Create individual detailed plots
    create_detailed_method_comparison(results, timestamp)
    create_accuracy_confidence_analysis(results, timestamp)
    
    plt.show()

def plot_overall_accuracy_comparison(results: List[Dict], ax):
    """Plot overall accuracy comparison across methods."""
    
    method_accuracy = {}
    method_counts = {}
    
    for result in results:
        for key, value in result.items():
            if key.endswith('_Accuracy') and value is not None:
                method_name = key.replace('_Accuracy', '')
                if method_name not in method_accuracy:
                    method_accuracy[method_name] = 0
                    method_counts[method_name] = 0
                if value:
                    method_accuracy[method_name] += 1
                method_counts[method_name] += 1
    
    # Calculate accuracy percentages
    methods = list(method_accuracy.keys())
    accuracies = [method_accuracy[m] / method_counts[m] * 100 if method_counts[m] > 0 else 0 for m in methods]
    
    # Create horizontal bar plot
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    bars = ax.barh(methods, accuracies, color=colors)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{acc:.1f}%', va='center', fontweight='bold')
    
    ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Overall Accuracy Comparison Across Methods', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3)

def plot_top_k_accuracy(results: List[Dict], ax):
    """Plot Top-K accuracy analysis."""
    
    methods = set()
    for result in results:
        for key in result.keys():
            if key.endswith('_Accuracy'):
                methods.add(key.replace('_Accuracy', ''))
    
    methods = list(methods)
    k_values = [1, 3, 5]
    
    accuracy_data = {method: [] for method in methods}
    
    for k in k_values:
        for method in methods:
            # Calculate top-k accuracy for each method
            correct = 0
            total = 0
            for result in results:
                if f'{method}_Accuracy' in result and result[f'{method}_Accuracy'] is not None:
                    total += 1
                    if result[f'{method}_Accuracy']:
                        correct += 1
            
            accuracy = correct / total * 100 if total > 0 else 0
            accuracy_data[method].append(accuracy)
    
    x = np.arange(len(k_values))
    width = 0.15
    
    for i, method in enumerate(methods):
        offset = (i - len(methods)//2) * width
        ax.bar(x + offset, accuracy_data[method], width, label=method)
    
    ax.set_xlabel('Top-K', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Top-K Accuracy Analysis', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Top-{k}' for k in k_values])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)

def plot_precision_recall(results: List[Dict], ax):
    """Plot Precision vs Recall scatter plot."""
    
    methods = set()
    for result in results:
        for key in result.keys():
            if key.endswith('_Accuracy'):
                methods.add(key.replace('_Accuracy', ''))
    
    for method in methods:
        precisions = []
        recalls = []
        
        for result in results:
            relevant_key = f'{method}_Relevant_Tables'
            count_key = f'{method}_Count'
            
            if relevant_key in result and count_key in result:
                relevant = result[relevant_key] or 0
                predicted = result[count_key] or 0
                
                # Estimate actual tables from ground truth (simplified)
                ground_truth = result.get('Ground_Truth_SQL', '')
                actual_tables = len(extract_tables_from_sql(ground_truth)) if ground_truth else 1
                
                precision = relevant / predicted if predicted > 0 else 0
                recall = relevant / actual_tables if actual_tables > 0 else 0
                
                precisions.append(precision)
                recalls.append(recall)
        
        if precisions and recalls:
            ax.scatter(recalls, precisions, label=method, alpha=0.7, s=60)
    
    ax.set_xlabel('Recall', fontweight='bold')
    ax.set_ylabel('Precision', fontweight='bold')
    ax.set_title('Precision vs Recall', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

def plot_performance_accuracy_tradeoff(results: List[Dict], ax):
    """Plot Performance vs Accuracy trade-off."""
    
    methods = set()
    for result in results:
        for key in result.keys():
            if key.endswith('_Duration_sec'):
                methods.add(key.replace('_Duration_sec', ''))
    
    for method in methods:
        durations = []
        accuracies = []
        
        for result in results:
            duration_key = f'{method}_Duration_sec'
            accuracy_key = f'{method}_Accuracy'
            
            if duration_key in result and accuracy_key in result:
                duration = result[duration_key] or 0
                accuracy = 1 if result[accuracy_key] else 0
                
                durations.append(duration)
                accuracies.append(accuracy)
        
        if durations and accuracies:
            avg_duration = np.mean(durations)
            avg_accuracy = np.mean(accuracies) * 100
            
            # Determine method type for coloring
            if any(llm in method for llm in ['GPT', 'Gemini', 'Azure']):
                color = 'red'
                marker = 'o'
            else:
                color = 'blue'
                marker = 's'
            
            ax.scatter(avg_duration, avg_accuracy, label=method, color=color, marker=marker, s=100)
            ax.annotate(method, (avg_duration, avg_accuracy), xytext=(5, 5), 
                       textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('Average Duration (seconds)', fontweight='bold')
    ax.set_ylabel('Average Accuracy (%)', fontweight='bold')
    ax.set_title('Performance vs Accuracy Trade-off', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add quadrant labels
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=np.median([np.mean([result[f'{m}_Duration_sec'] for result in results 
                                   if f'{m}_Duration_sec' in result]) 
                           for m in methods]), color='gray', linestyle='--', alpha=0.5)

def plot_confidence_distribution(results: List[Dict], ax):
    """Plot confidence score distribution."""
    
    methods = set()
    for result in results:
        for key in result.keys():
            if key.endswith('_Confidence'):
                methods.add(key.replace('_Confidence', ''))
    
    confidence_data = {method: [] for method in methods}
    
    for result in results:
        for method in methods:
            conf_key = f'{method}_Confidence'
            if conf_key in result and result[conf_key] is not None:
                confidence_data[method].append(result[conf_key])
    
    # Create violin plot
    data_to_plot = [confidence_data[method] for method in methods if confidence_data[method]]
    if data_to_plot:
        parts = ax.violinplot(data_to_plot, positions=range(len(data_to_plot)), showmeans=True)
        ax.set_xticks(range(len(data_to_plot)))
        ax.set_xticklabels([m for m in methods if confidence_data[m]], rotation=45)
    
    ax.set_ylabel('Confidence Score', fontweight='bold')
    ax.set_title('Confidence Score Distribution', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

def plot_method_comparison_heatmap(results: List[Dict], ax):
    """Create method comparison heatmap."""
    
    methods = list(set(key.replace('_Accuracy', '') for result in results 
                      for key in result.keys() if key.endswith('_Accuracy')))
    
    metrics = ['Accuracy', 'Confidence', 'Count', 'Duration_sec']
    
    heatmap_data = np.zeros((len(methods), len(metrics)))
    
    for i, method in enumerate(methods):
        for j, metric in enumerate(metrics):
            values = []
            for result in results:
                key = f'{method}_{metric}'
                if key in result and result[key] is not None:
                    if metric == 'Accuracy':
                        values.append(1 if result[key] else 0)
                    else:
                        values.append(result[key])
            
            if values:
                if metric == 'Duration_sec':
                    heatmap_data[i, j] = 1 / (np.mean(values) + 0.001)  # Inverse for duration
                else:
                    heatmap_data[i, j] = np.mean(values)
    
    # Normalize data for better visualization
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    heatmap_data_normalized = scaler.fit_transform(heatmap_data)
    
    im = ax.imshow(heatmap_data_normalized, cmap='RdYlBu_r', aspect='auto')
    
    ax.set_xticks(range(len(metrics)))
    ax.set_yticks(range(len(methods)))
    ax.set_xticklabels(metrics)
    ax.set_yticklabels(methods)
    ax.set_title('Method Performance Heatmap', fontweight='bold')
    
    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{heatmap_data_normalized[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax, shrink=0.8)

def plot_query_difficulty_analysis(results: List[Dict], ax):
    """Analyze query difficulty vs accuracy."""
    
    # Use query length as a proxy for difficulty
    query_lengths = []
    avg_accuracies = []
    
    for result in results:
        length = result.get('Question_Length', 0)
        
        # Calculate average accuracy across all methods for this query
        accuracies = []
        for key, value in result.items():
            if key.endswith('_Accuracy') and value is not None:
                accuracies.append(1 if value else 0)
        
        if accuracies:
            query_lengths.append(length)
            avg_accuracies.append(np.mean(accuracies) * 100)
    
    # Create scatter plot with trend line
    if query_lengths and avg_accuracies:
        ax.scatter(query_lengths, avg_accuracies, alpha=0.6, s=50)
        
        # Add trend line
        z = np.polyfit(query_lengths, avg_accuracies, 1)
        p = np.poly1d(z)
        ax.plot(sorted(query_lengths), p(sorted(query_lengths)), "r--", alpha=0.8)
        
        # Add correlation coefficient
        corr = np.corrcoef(query_lengths, avg_accuracies)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    ax.set_xlabel('Query Length (characters)', fontweight='bold')
    ax.set_ylabel('Average Accuracy (%)', fontweight='bold')
    ax.set_title('Query Difficulty vs Accuracy', fontweight='bold')
    ax.grid(True, alpha=0.3)

def create_detailed_method_comparison(results: List[Dict], timestamp: str):
    """Create detailed method comparison plots."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Success Rate by Method
    plot_success_rate_by_method(results, ax1)
    
    # 2. Average Tables Found by Method
    plot_avg_tables_by_method(results, ax2)
    
    # 3. Response Time Distribution
    plot_response_time_distribution(results, ax3)
    
    # 4. Accuracy vs Query Type Analysis
    plot_accuracy_by_query_type(results, ax4)
    
    plt.suptitle('Detailed Method Comparison Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = f"detailed_method_comparison_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Detailed method comparison saved to {filename}")
    plt.show()

def plot_success_rate_by_method(results: List[Dict], ax):
    """Plot success rate by method."""
    
    methods = set()
    for result in results:
        for key in result.keys():
            if key.endswith('_Count'):
                methods.add(key.replace('_Count', ''))
    
    success_rates = {}
    for method in methods:
        successful = sum(1 for result in results 
                        if result.get(f'{method}_Count', 0) > 0)
        total = sum(1 for result in results 
                   if f'{method}_Count' in result)
        success_rates[method] = (successful / total * 100) if total > 0 else 0
    
    methods_sorted = sorted(success_rates.keys(), key=lambda x: success_rates[x], reverse=True)
    rates = [success_rates[m] for m in methods_sorted]
    
    colors = ['#2E8B57' if rate > 80 else '#FF6347' if rate < 50 else '#FFD700' for rate in rates]
    bars = ax.bar(methods_sorted, rates, color=colors)
    
    # Add value labels
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Success Rate (%)', fontweight='bold')
    ax.set_title('Success Rate by Method', fontweight='bold')
    ax.set_ylim(0, 105)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

def plot_avg_tables_by_method(results: List[Dict], ax):
    """Plot average number of tables found by method."""
    
    methods = set()
    for result in results:
        for key in result.keys():
            if key.endswith('_Count'):
                methods.add(key.replace('_Count', ''))
    
    avg_tables = {}
    for method in methods:
        counts = [result.get(f'{method}_Count', 0) for result in results 
                 if f'{method}_Count' in result]
        avg_tables[method] = np.mean(counts) if counts else 0
    
    methods_sorted = sorted(avg_tables.keys(), key=lambda x: avg_tables[x], reverse=True)
    avgs = [avg_tables[m] for m in methods_sorted]
    
    bars = ax.bar(methods_sorted, avgs, color='skyblue', edgecolor='navy', linewidth=1)
    
    # Add value labels
    for bar, avg in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{avg:.1f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Average Tables Found', fontweight='bold')
    ax.set_title('Average Tables Found by Method', fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

def plot_response_time_distribution(results: List[Dict], ax):
    """Plot response time distribution."""
    
    methods = set()
    for result in results:
        for key in result.keys():
            if key.endswith('_Duration_sec'):
                methods.add(key.replace('_Duration_sec', ''))
    
    duration_data = {method: [] for method in methods}
    
    for result in results:
        for method in methods:
            duration_key = f'{method}_Duration_sec'
            if duration_key in result and result[duration_key] is not None:
                duration_data[method].append(result[duration_key])
    
    # Create box plot
    data_to_plot = [duration_data[method] for method in methods if duration_data[method]]
    method_labels = [method for method in methods if duration_data[method]]
    
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=method_labels, patch_artist=True)
        
        # Color boxes
        colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink', 'lightgray']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
    
    ax.set_ylabel('Response Time (seconds)', fontweight='bold')
    ax.set_title('Response Time Distribution', fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

def plot_accuracy_by_query_type(results: List[Dict], ax):
    """Plot accuracy by query type (categorized by keywords)."""
    
    # Define query categories based on keywords
    categories = {
        'Trade Queries': ['trade', 'trading', 'etd', 'execution'],
        'Product Queries': ['product', 'cusip', 'instrument', 'security'],
        'Entity Queries': ['counterparty', 'trader', 'entity', 'government'],
        'Time-based Queries': ['yesterday', 'last week', 'date', 'time'],
        'Currency Queries': ['currency', 'exchange', 'rate'],
        'Other': []
    }
    
    category_accuracies = {cat: [] for cat in categories.keys()}
    
    for result in results:
        question = result.get('Question', '').lower()
        
        # Categorize the query
        assigned_category = 'Other'
        for category, keywords in categories.items():
            if any(keyword in question for keyword in keywords):
                assigned_category = category
                break
        
        # Calculate average accuracy for this query
        accuracies = []
        for key, value in result.items():
            if key.endswith('_Accuracy') and value is not None:
                accuracies.append(1 if value else 0)
        
        if accuracies:
            category_accuracies[assigned_category].append(np.mean(accuracies) * 100)
    
    # Plot as grouped bar chart
    categories_with_data = {cat: accs for cat, accs in category_accuracies.items() if accs}
    
    if categories_with_data:
        cat_names = list(categories_with_data.keys())
        avg_accs = [np.mean(categories_with_data[cat]) for cat in cat_names]
        std_accs = [np.std(categories_with_data[cat]) for cat in cat_names]
        
        bars = ax.bar(cat_names, avg_accs, yerr=std_accs, capsize=5, 
                     color='lightcoral', edgecolor='darkred', linewidth=1)
        
        # Add value labels
        for bar, avg in zip(bars, avg_accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{avg:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Average Accuracy (%)', fontweight='bold')
    ax.set_title('Accuracy by Query Category', fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

def create_accuracy_confidence_analysis(results: List[Dict], timestamp: str):
    """Create accuracy vs confidence analysis."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Accuracy vs Confidence Scatter
    plot_accuracy_confidence_scatter(results, ax1)
    
    # 2. Calibration Plot
    plot_calibration_curve(results, ax2)
    
    # 3. Method Reliability Analysis
    plot_method_reliability(results, ax3)
    
    # 4. Error Analysis
    plot_error_analysis(results, ax4)
    
    plt.suptitle('Accuracy and Confidence Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = f"accuracy_confidence_analysis_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Accuracy confidence analysis saved to {filename}")
    plt.show()

def plot_accuracy_confidence_scatter(results: List[Dict], ax):
    """Plot accuracy vs confidence scatter plot."""
    
    methods = set()
    for result in results:
        for key in result.keys():
            if key.endswith('_Accuracy'):
                methods.add(key.replace('_Accuracy', ''))
    
    for method in methods:
        confidences = []
        accuracies = []
        
        for result in results:
            conf_key = f'{method}_Confidence'
            acc_key = f'{method}_Accuracy'
            
            if (conf_key in result and acc_key in result and 
                result[conf_key] is not None and result[acc_key] is not None):
                confidences.append(result[conf_key])
                accuracies.append(1 if result[acc_key] else 0)
        
        if confidences and accuracies:
            ax.scatter(confidences, accuracies, label=method, alpha=0.7, s=40)
    
    ax.set_xlabel('Confidence Score', fontweight='bold')
    ax.set_ylabel('Accuracy (0/1)', fontweight='bold')
    ax.set_title('Accuracy vs Confidence', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add diagonal line for perfect calibration
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.8, label='Perfect Calibration')

def plot_calibration_curve(results: List[Dict], ax):
    """Plot calibration curve for confidence scores."""
    
    # Aggregate all confidence and accuracy pairs
    all_confidences = []
    all_accuracies = []
    
    for result in results:
        for key, value in result.items():
            if key.endswith('_Confidence') and value is not None:
                method = key.replace('_Confidence', '')
                acc_key = f'{method}_Accuracy'
                if acc_key in result and result[acc_key] is not None:
                    all_confidences.append(value)
                    all_accuracies.append(1 if result[acc_key] else 0)
    
    if all_confidences and all_accuracies:
        # Bin the confidence scores
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = [(conf >= bin_lower) and (conf < bin_upper) 
                     for conf in all_confidences]
            if any(in_bin):
                bin_acc = np.mean([acc for acc, in_b in zip(all_accuracies, in_bin) if in_b])
                bin_conf = np.mean([conf for conf, in_b in zip(all_confidences, in_bin) if in_b])
                bin_count = sum(in_bin)
                
                bin_accuracies.append(bin_acc)
                bin_confidences.append(bin_conf)
                bin_counts.append(bin_count)
        
        # Plot calibration curve
        ax.plot(bin_confidences, bin_accuracies, 'o-', linewidth=2, markersize=8)
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.8, label='Perfect Calibration')
        
        # Add bin counts as annotations
        for conf, acc, count in zip(bin_confidences, bin_accuracies, bin_counts):
            ax.annotate(f'n={count}', (conf, acc), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Mean Predicted Confidence', fontweight='bold')
    ax.set_ylabel('Fraction of Positives', fontweight='bold')
    ax.set_title('Calibration Curve', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

def plot_method_reliability(results: List[Dict], ax):
    """Plot method reliability analysis."""
    
    methods = set()
    for result in results:
        for key in result.keys():
            if key.endswith('_Accuracy'):
                methods.add(key.replace('_Accuracy', ''))
    
    reliability_scores = {}
    
    for method in methods:
        accuracies = []
        confidences = []
        
        for result in results:
            acc_key = f'{method}_Accuracy'
            conf_key = f'{method}_Confidence'
            
            if (acc_key in result and conf_key in result and 
                result[acc_key] is not None and result[conf_key] is not None):
                accuracies.append(1 if result[acc_key] else 0)
                confidences.append(result[conf_key])
        
        if accuracies and confidences:
            # Calculate reliability as correlation between confidence and accuracy
            if len(set(accuracies)) > 1 and len(set(confidences)) > 1:
                reliability = np.corrcoef(confidences, accuracies)[0, 1]
            else:
                reliability = 0
            reliability_scores[method] = reliability
    
    if reliability_scores:
        methods_sorted = sorted(reliability_scores.keys(), 
                               key=lambda x: reliability_scores[x], reverse=True)
        scores = [reliability_scores[m] for m in methods_sorted]
        
        colors = ['green' if score > 0.3 else 'orange' if score > 0 else 'red' for score in scores]
        bars = ax.bar(methods_sorted, scores, color=colors)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, 
                   bar.get_height() + (0.02 if score >= 0 else -0.05),
                   f'{score:.3f}', ha='center', 
                   va='bottom' if score >= 0 else 'top', fontweight='bold')
    
    ax.set_ylabel('Reliability Score (Confidence-Accuracy Correlation)', fontweight='bold')
    ax.set_title('Method Reliability Analysis', fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)

def plot_error_analysis(results: List[Dict], ax):
    """Plot error analysis by method."""
    
    methods = set()
    for result in results:
        for key in result.keys():
            if key.endswith('_Accuracy'):
                methods.add(key.replace('_Accuracy', ''))
    
    error_types = {'False Positive': [], 'False Negative': [], 'True Positive': [], 'True Negative': []}
    method_names = []
    
    for method in methods:
        tp = fp = tn = fn = 0
        
        for result in results:
            acc_key = f'{method}_Accuracy'
            count_key = f'{method}_Count'
            
            if acc_key in result and count_key in result:
                predicted_count = result[count_key] or 0
                is_accurate = result[acc_key] if result[acc_key] is not None else False
                
                if predicted_count > 0 and is_accurate:
                    tp += 1
                elif predicted_count > 0 and not is_accurate:
                    fp += 1
                elif predicted_count == 0 and is_accurate:
                    tn += 1
                elif predicted_count == 0 and not is_accurate:
                    fn += 1
        
        total = tp + fp + tn + fn
        if total > 0:
            error_types['True Positive'].append(tp / total * 100)
            error_types['False Positive'].append(fp / total * 100)
            error_types['True Negative'].append(tn / total * 100)
            error_types['False Negative'].append(fn / total * 100)
            method_names.append(method)
    
    if method_names:
        x = np.arange(len(method_names))
        width = 0.6
        
        bottom = np.zeros(len(method_names))
        colors = ['green', 'red', 'lightgreen', 'orange']
        
        for (error_type, values), color in zip(error_types.items(), colors):
            ax.bar(x, values, width, bottom=bottom, label=error_type, color=color)
            bottom += values
    
    ax.set_ylabel('Percentage (%)', fontweight='bold')
    ax.set_title('Error Analysis by Method', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)