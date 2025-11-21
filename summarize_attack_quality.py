#!/usr/bin/env python3
"""
Comprehensive summary of attack quality and training implications.
"""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Summarize attack quality")
    parser.add_argument("--input", type=str, required=True, help="Parsed evaluation CSV")
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.input)
    
    print("="*80)
    print("ATTACK QUALITY ASSESSMENT")
    print("="*80)
    
    # Overall stats
    print(f"\nTotal attacks evaluated: {len(df)}")
    
    # Plausibility
    print("\n--- MEDICAL PLAUSIBILITY ---")
    plausible_pct = (df['plausibility'] == 'plausible').sum() / len(df) * 100
    implausible_pct = (df['plausibility'] == 'implausible').sum() / len(df) * 100
    print(f"Plausible: {(df['plausibility'] == 'plausible').sum()} ({plausible_pct:.1f}%)")
    print(f"Implausible: {(df['plausibility'] == 'implausible').sum()} ({implausible_pct:.1f}%)")
    
    if implausible_pct > 30:
        print("⚠️  HIGH: >30% implausible attacks - training on unrealistic errors")
    elif implausible_pct > 15:
        print("⚠️  MODERATE: 15-30% implausible - some noise in training data")
    else:
        print("✓ GOOD: <15% implausible attacks")
    
    # Difficulty
    print("\n--- DETECTION DIFFICULTY ---")
    print(df['difficulty'].value_counts())
    
    easy_pct = (df['difficulty'] == 'easy').sum() / len(df) * 100
    hard_pct = (df['difficulty'] == 'hard').sum() / len(df) * 100
    
    if easy_pct > 50:
        print("⚠️  Too many EASY attacks - assessor won't learn much")
    if hard_pct < 10:
        print("⚠️  Too few HARD attacks - not challenging enough")
    
    # Assessor performance
    if 'assessor_correct' in df.columns:
        print("\n--- ASSESSOR PERFORMANCE ---")
        overall_acc = df['assessor_correct'].mean() * 100
        print(f"Overall accuracy: {overall_acc:.1f}%")
        
        # Filter to plausible attacks only
        plausible_df = df[df['plausibility'] == 'plausible']
        if len(plausible_df) > 0:
            plausible_acc = plausible_df['assessor_correct'].mean() * 100
            print(f"Accuracy on PLAUSIBLE attacks: {plausible_acc:.1f}%")
            
            if plausible_acc > 80:
                print("⚠️  CRITICAL: Assessor >80% accurate on plausible attacks")
                print("   → Attacks are TOO EASY - assessor has little to learn")
            elif plausible_acc > 70:
                print("⚠️  WARNING: Assessor 70-80% accurate")
                print("   → Attacks could be more challenging")
            elif plausible_acc < 50:
                print("⚠️  WARNING: Assessor <50% accurate")
                print("   → Attacks might be too hard or task is unrealistic")
            else:
                print("✓ GOOD: Assessor 50-70% accurate - appropriate difficulty")
        
        # Implausible attacks
        implausible_df = df[df['plausibility'] == 'implausible']
        if len(implausible_df) > 0:
            implausible_acc = implausible_df['assessor_correct'].mean() * 100
            print(f"Accuracy on IMPLAUSIBLE attacks: {implausible_acc:.1f}%")
            print("   (Lower is expected - these are bad training examples)")
    
    # Clinical impact
    print("\n--- CLINICAL IMPACT ---")
    print(df['impact'].value_counts())
    
    severe_pct = (df['impact'] == 'severe').sum() / len(df) * 100
    minor_pct = (df['impact'] == 'minor').sum() / len(df) * 100
    
    if severe_pct < 20:
        print("⚠️  Few SEVERE impact attacks - might want more high-stakes errors")
    if minor_pct > 50:
        print("⚠️  Many MINOR impact attacks - low clinical relevance")
    
    # Detailed breakdown
    print("\n--- DETAILED BREAKDOWN ---")
    print("\nPlausible attacks by difficulty:")
    plausible_by_diff = df[df['plausibility'] == 'plausible'].groupby('difficulty').size()
    print(plausible_by_diff)
    
    if 'assessor_correct' in df.columns:
        print("\nAssessor accuracy by (plausibility, difficulty):")
        breakdown = df.groupby(['plausibility', 'difficulty'])['assessor_correct'].agg(['mean', 'count'])
        print(breakdown)
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if implausible_pct > 20:
        print("\n1. REDUCE IMPLAUSIBLE ATTACKS:")
        print("   - Add medical plausibility constraint to attacker reward")
        print("   - Use few-shot examples of realistic medical errors")
        print("   - Lower attacker temperature for more conservative edits")
    
    if 'assessor_correct' in df.columns and plausible_acc > 75:
        print("\n2. INCREASE ATTACK DIFFICULTY:")
        print("   - Reward more subtle, harder-to-detect errors")
        print("   - Penalize obvious contradictions")
        print("   - Use curriculum learning: start easy, increase difficulty")
        print("   - Add diversity bonus to avoid repetitive attack patterns")
    
    if easy_pct > 40:
        print("\n3. REDUCE EASY ATTACKS:")
        print("   - Filter out attacks with obvious errors during training")
        print("   - Increase minimum edit distance requirements")
        print("   - Reward attacks that fool the assessor more")
    
    if 'total_reward' in df.columns:
        print("\n4. REWARD FUNCTION ANALYSIS:")
        high_reward = df[df['total_reward'] > df['total_reward'].quantile(0.75)]
        if len(high_reward) > 0:
            high_reward_plausible = (high_reward['plausibility'] == 'plausible').mean() * 100
            print(f"   High-reward attacks that are plausible: {high_reward_plausible:.1f}%")
            if high_reward_plausible < 70:
                print("   ⚠️  Reward function may be incentivizing implausible attacks")
    
    print("\n5. CONSIDER ALTERNATIVE APPROACHES:")
    print("   - Use medgemma-4b as a plausibility filter during training")
    print("   - Pre-train attacker on MEDEC dataset for realistic errors")
    print("   - Add adversarial training with difficulty curriculum")
    print("   - Use rejection sampling to keep only plausible, medium/hard attacks")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
