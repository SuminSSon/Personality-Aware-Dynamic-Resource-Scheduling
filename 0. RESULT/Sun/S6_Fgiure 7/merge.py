import pandas as pd
import os
import glob

def merge_logs():
    # 1. 설정: 노드 리스트
    nodes_list = [64, 128, 256, 512, 1024, 2048, 4096]
    
    # [수정] 실제 생성된 파일명에 맞춰 스케줄러 이름 수정 ('Ours' -> 'Ours (mixed)')
    schedulers = ["FIFO", "SFJ", "Pollux", "Sia", "Lucid", "Ours (mixed)", "Ours_Cost", "Ours_Time"]
    
    # 집계할 컬럼 (simulation.py 출력 기준)
    columns_to_aggregate = ['JCT', 'Alloc_GPUs', 'Energy_kWh', 'Cost_$']

    summary_data = [] 
    
    print("🔄 Starting Merge Process...")

    # 2. 스케줄러별 파일 순회 및 병합
    for scheduler in schedulers:
        scheduler_dfs = []
        print(f"  > Processing scheduler: {scheduler}")
        
        for node in nodes_list:
            # 파일명 패턴 매칭
            # 주의: 파일명에 괄호나 공백이 있을 수 있으므로 glob 패턴에 유의
            # run.py가 'details_64_Ours (mixed).csv' 형태로 저장했다고 가정
            
            search_pattern = f"**/details_{node}_{scheduler}.csv"
            # 혹시 모르니 glob이 특수문자를 잘 처리하도록 escape가 필요할 수도 있으나, 
            # 일단 python glob은 대괄호[] 외에는 리터럴로 처리하므로 시도
            files = glob.glob(search_pattern, recursive=True)
            
            if not files:
                continue
                
            file_name = files[0]
            
            try:
                df = pd.read_csv(file_name)
                
                # 병합용 데이터에 노드 정보 추가
                df['Nodes'] = node
                df['Scheduler'] = scheduler
                scheduler_dfs.append(df)
                
                # 요약 통계 계산
                row_data = {
                    'Scheduler': scheduler,
                    'Nodes': node
                }
                
                for col in columns_to_aggregate:
                    if col in df.columns:
                        row_data[f'{col}_mean'] = df[col].mean()
                        row_data[f'{col}_sum'] = df[col].sum()
                        row_data[f'{col}_max'] = df[col].max()
                
                summary_data.append(row_data)
                
            except Exception as e:
                print(f"    ⚠️ Error reading {file_name}: {e}")

        # 스케줄러별 병합 파일 저장
        if scheduler_dfs:
            merged_df = pd.concat(scheduler_dfs, ignore_index=True)
            # 파일 저장 시에는 공백이나 괄호가 있어도 상관없음
            safe_sched_name = scheduler.replace(" ", "_").replace("(", "").replace(")", "")
            output_filename = f"{safe_sched_name}_merged.csv"
            merged_df.to_csv(output_filename, index=False)
            print(f"    ✅ Saved merged file: {output_filename}")
        else:
            print(f"    ℹ️ No data found for {scheduler}")

    # 3. 전체 요약 파일 저장
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_output_csv = 'all_combinations_summary.csv'
        summary_df.to_csv(summary_output_csv, index=False)
        print(f"\n🎉 Summary file saved as: {summary_output_csv}")
    else:
        print("\n⚠️ No data found to merge.")

if __name__ == "__main__":
    merge_logs()