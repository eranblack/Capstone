def cross_val(df,number=5):
    score=[]
    team_resids_bench=[]
    team_resids_pred=[]
    for element in range(number):
        cross_model=ModelP(2008+element)
        cross_model.fit(df)
        y_pred=cross_model.predict()
        score.append(cross_model.score())

        team_pred=cross_model.teams()
        team_resids_bench.append(team_pred['mean_bencmark_resids'])
        team_resids_pred.append(team_pred['mean_pred_resids'])
    return (np.mean(score),np.mean(team_resids_bench),np.mean(team_resids_pred)) 
