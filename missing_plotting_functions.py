# ADD THESE MISSING PLOTTING FUNCTIONS TO YOUR SCRIPT:

def plot_comprehensive_radar_chart(summary_df: pd.DataFrame, ax):
    """Plot comprehensive radar chart for method comparison."""
    
    try:
        methods = summary_df['Method'].tolist()[:6]  # Limit to 6 methods for readability
        
        # Metrics for radar chart
        metrics = ['Precision@1_%', 'Precision@5_%', 'Recall@5_%', 'Recall@10_%', 'Avg_Coverage_%', 'Retrieval_Success_%']
        metric_labels = ['P@1', 'P@5', 'R@5', 'R@10', 'Coverage', 'Success']
        
        # Number of metrics
        N = len(metrics)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Plot for each method
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
        
        for i, method in enumerate(methods):
            method_row = summary_df[summary_df['Method'] == method].iloc[0]
            
            # Get values for this method
            values = []
            for metric in metrics:
                if metric in method_row:
                    val = method_row[metric]
                    # Normalize to 0-100 scale
                    if metric == 'Avg_Coverage_%':
                        values.append(val)  # Already percentage
                    else:
                        values.append(val)  # Already percentage
                else:
                    values.append(0)
            
            values += values[:1]  # Complete the circle
            
            # Plot
            ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 100)
        ax.set_title('Method Performance Radar Chart', fontweight='bold')
        ax.legend(bbox_to_anchor=(1.1, 1.1))
        ax.grid(True)
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Radar chart error: {str(e)}', transform=ax.transAxes, ha='center')

def plot_accuracy_type_heatmap(summary_df: pd.DataFrame, ax):
    """Plot heatmap showing different accuracy types."""
    
    try:
        methods = summary_df['Method'].tolist()
        accuracy_types = ['Precision@1_%', 'Precision@3_%', 'Precision@5_%', 'Recall@3_%', 'Recall@5_%', 'Recall@10_%']
        type_labels = ['P@1', 'P@3', 'P@5', 'R@3', 'R@5', 'R@10']
        
        # Create data matrix
        data_matrix = []
        for method in methods:
            method_row = summary_df[summary_df['Method'] == method].iloc[0]
            row_data = []
            for acc_type in accuracy_types:
                if acc_type in method_row:
                    row_data.append(method_row[acc_type])
                else:
                    row_data.append(0)
            data_matrix.append(row_data)
        
        data_matrix = np.array(data_matrix)
        
        # Create heatmap
        im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        
        # Set ticks and labels
        ax.set_xticks(range(len(type_labels)))
        ax.set_yticks(range(len(methods)))
        ax.set_xticklabels(type_labels)
        ax.set_yticklabels(methods)
        
        # Add text annotations
        for i in range(len(methods)):
            for j in range(len(type_labels)):
                text = ax.text(j, i, f'{data_matrix[i, j]:.0f}%',
                              ha="center", va="center", color="black", fontsize=8, fontweight='bold')
        
        ax.set_title('Accuracy Types Heatmap', fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.6)
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Heatmap error: {str(e)}', transform=ax.transAxes, ha='center')

def plot_recall_progression(summary_df: pd.DataFrame, ax):
    """Plot recall progression from R@3 to R@10."""
    
    try:
        methods = summary_df['Method'].tolist()
        recall_3 = summary_df['Recall@3_%'].tolist()
        recall_5 = summary_df['Recall@5_%'].tolist()
        recall_10 = summary_df['Recall@10_%'].tolist()
        
        x = np.arange(len(methods))
        
        # Plot lines for each method
        for i, method in enumerate(methods):
            recalls = [recall_3[i], recall_5[i], recall_10[i]]
            k_values = [3, 5, 10]
            ax.plot(k_values, recalls, 'o-', label=method, linewidth=2, markersize=6)
        
        ax.set_xlabel('K (Top-K Results)', fontweight='bold')
        ax.set_ylabel('Recall@K (%)', fontweight='bold')
        ax.set_title('Recall@K Progression\n(Higher = Better for Graph Methods)', fontweight='bold')
        ax.set_xticks([3, 5, 10])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Recall progression error: {str(e)}', transform=ax.transAxes, ha='center')

def plot_success_rate_analysis(summary_df: pd.DataFrame, ax):
    """Plot success rate analysis."""
    
    try:
        methods = summary_df['Method'].tolist()
        success_rates = summary_df['Retrieval_Success_%'].tolist()
        query_success = summary_df['Query_Success_Rate_%'].tolist()
        
        # Create scatter plot
        colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
        
        for i, method in enumerate(methods):
            ax.scatter(query_success[i], success_rates[i], 
                      s=100, c=[colors[i]], label=method, alpha=0.7)
            ax.annotate(method, (query_success[i], success_rates[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('Query Success Rate (%)', fontweight='bold')
        ax.set_ylabel('Retrieval Success Rate (%)', fontweight='bold')
        ax.set_title('Success Rate Analysis', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
        
        # Add diagonal reference line
        ax.plot([0, 100], [0, 100], 'r--', alpha=0.5, label='Perfect Correlation')
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Success rate error: {str(e)}', transform=ax.transAxes, ha='center')

def plot_coverage_speed_tradeoff(summary_df: pd.DataFrame, ax):
    """Plot coverage vs speed trade-off."""
    
    try:
        methods = summary_df['Method'].tolist()
        coverage = summary_df['Avg_Coverage_%'].tolist()
        duration = summary_df['Avg_Duration_sec'].tolist()
        
        # Create scatter plot with size based on success rate
        success_rates = summary_df['Retrieval_Success_%'].tolist()
        sizes = [rate * 3 for rate in success_rates]  # Scale for visibility
        
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(methods)))
        
        for i, method in enumerate(methods):
            ax.scatter(duration[i], coverage[i], s=sizes[i], 
                      c=[colors[i]], label=method, alpha=0.7, edgecolors='black')
            ax.annotate(method, (duration[i], coverage[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('Average Duration (seconds)', fontweight='bold')
        ax.set_ylabel('Average Coverage (%)', fontweight='bold')
        ax.set_title('Coverage vs Speed Trade-off\n(Bubble size = Success Rate)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add quadrant labels
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=np.median(duration), color='gray', linestyle='--', alpha=0.5)
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Coverage-speed error: {str(e)}', transform=ax.transAxes, ha='center')

def plot_method_recommendation_matrix(summary_df: pd.DataFrame, ax):
    """Plot method recommendation matrix."""
    
    try:
        methods = summary_df['Method'].tolist()
        precision_1 = summary_df['Precision@1_%'].tolist()
        recall_5 = summary_df['Recall@5_%'].tolist()
        
        # Create quadrant plot
        colors = []
        recommendations = []
        
        for i, method in enumerate(methods):
            p1 = precision_1[i]
            r5 = recall_5[i]
            
            if p1 >= 60 and r5 >= 80:
                color = 'green'
                rec = 'Excellent All-Round'
            elif p1 >= 60:
                color = 'blue'
                rec = 'Precision Leader'
            elif r5 >= 80:
                color = 'orange'
                rec = 'Recall Leader'
            else:
                color = 'red'
                rec = 'Needs Improvement'
            
            colors.append(color)
            recommendations.append(rec)
        
        # Plot points
        for i, method in enumerate(methods):
            ax.scatter(recall_5[i], precision_1[i], s=150, 
                      c=colors[i], label=recommendations[i], alpha=0.7)
            ax.annotate(method, (recall_5[i], precision_1[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Recall@5 (%) - Graph Method Strength', fontweight='bold')
        ax.set_ylabel('Precision@1 (%) - Ranking Strength', fontweight='bold')
        ax.set_title('Method Recommendation Matrix', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add quadrant lines
        ax.axhline(y=60, color='gray', linestyle='--', alpha=0.7, label='Precision Threshold')
        ax.axvline(x=80, color='gray', linestyle='--', alpha=0.7, label='Recall Threshold')
        
        # Remove duplicate labels in legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Recommendation matrix error: {str(e)}', transform=ax.transAxes, ha='center')

def plot_accuracy_improvements(summary_df: pd.DataFrame, ax):
    """Plot accuracy improvements from strict to relaxed metrics."""
    
    try:
        methods = summary_df['Method'].tolist()
        
        # Calculate improvements
        p1_to_p5_improvement = []
        r3_to_r10_improvement = []
        
        for method in methods:
            method_row = summary_df[summary_df['Method'] == method].iloc[0]
            
            p1 = method_row.get('Precision@1_%', 0)
            p5 = method_row.get('Precision@5_%', 0)
            r3 = method_row.get('Recall@3_%', 0)
            r10 = method_row.get('Recall@10_%', 0)
            
            p_improvement = p5 - p1 if p1 > 0 else 0
            r_improvement = r10 - r3 if r3 > 0 else 0
            
            p1_to_p5_improvement.append(p_improvement)
            r3_to_r10_improvement.append(r_improvement)
        
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, p1_to_p5_improvement, width, 
                       label='Precision@1→5 Gain', color='#FF7F7F', alpha=0.8)
        bars2 = ax.bar(x + width/2, r3_to_r10_improvement, width,
                       label='Recall@3→10 Gain', color='#7FFF7F', alpha=0.8)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'+{height:.0f}%', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'+{height:.0f}%', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Methods', fontweight='bold')
        ax.set_ylabel('Accuracy Improvement (%)', fontweight='bold')
        ax.set_title('Accuracy Gains: Strict → Relaxed Metrics', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Improvements error: {str(e)}', transform=ax.transAxes, ha='center')

def plot_overall_performance_score(summary_df: pd.DataFrame, ax):
    """Plot overall performance score combining multiple metrics."""
    
    try:
        methods = summary_df['Method'].tolist()
        
        # Calculate composite performance score
        performance_scores = []
        
        for method in methods:
            method_row = summary_df[summary_df['Method'] == method].iloc[0]
            
            # Weighted score: 30% Precision@1, 40% Recall@5, 20% Coverage, 10% Speed
            p1 = method_row.get('Precision@1_%', 0)
            r5 = method_row.get('Recall@5_%', 0)
            cov = method_row.get('Avg_Coverage_%', 0)
            duration = method_row.get('Avg_Duration_sec', 1)
            
            # Speed score (inverse of duration, normalized)
            max_duration = summary_df['Avg_Duration_sec'].max()
            speed_score = (1 - (duration / max_duration)) * 100 if max_duration > 0 else 50
            
            # Composite score
            score = (0.3 * p1 + 0.4 * r5 + 0.2 * cov + 0.1 * speed_score)
            performance_scores.append(score)
        
        # Sort by performance score
        sorted_data = sorted(zip(methods, performance_scores), key=lambda x: x[1], reverse=True)
        sorted_methods, sorted_scores = zip(*sorted_data)
        
        # Create horizontal bar chart
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_methods)))
        bars = ax.barh(sorted_methods, sorted_scores, color=colors)
        
        # Add value labels
        for bar, score in zip(bars, sorted_scores):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{score:.1f}', va='center', fontweight='bold')
        
        ax.set_xlabel('Overall Performance Score', fontweight='bold')
        ax.set_title('Overall Performance Ranking\n(30% P@1 + 40% R@5 + 20% Coverage + 10% Speed)', 
                     fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(0, max(sorted_scores) * 1.1)
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Performance score error: {str(e)}', transform=ax.transAxes, ha='center')