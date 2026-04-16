import pandas as pd
import plotly.express as px
import streamlit as st
import os

# PHASE 2 — DATA WRANGLING FUNCTIONS
# Handling all CSV loading, column cleaning, and dataframe construction.

@st.cache_data
def loadAllData(dataFolder="."):
    """
    loading all 7 season CSV files, tagging each with a Season column,
    and combining them into one unified dataframe.
    """
    seasonFiles = [
        "2019.csv", "2020.csv", "2021.csv",
        "2022.csv", "2023.csv", "2024.csv", "2025.csv"
    ]
    allFrames = []

    for fileName in seasonFiles:
        filePath = os.path.join(dataFolder, fileName)

        # checking if the file actually exists before attempting to load
        if not os.path.exists(filePath):
            continue

        # reading the CSV into a dataframe
        df = pd.read_csv(filePath)

        # extracting the season year from the filename
        seasonYear = int(fileName.replace(".csv", ""))

        # tagging every row with its corresponding season
        df["Season"] = seasonYear

        allFrames.append(df)

    # combining all the individual season frames into one master dataframe
    if len(allFrames) == 0:
        st.error("No CSV files found. Please place your season CSVs in the app directory.")
        st.stop()

    combinedDf = pd.concat(allFrames, ignore_index=True)
    return combinedDf


def cleanData(rawDf):
    """
    cleaning the raw combined dataframe by converting position and grid columns
    to numeric types, and flagging non-finishers (DNFs) with a separate column.
    """
    df = rawDf.copy()

    # stripping any accidental whitespace from string columns
    stringCols = ["Driver", "Team", "Track", "Fastest Lap"]
    for col in stringCols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # converting Position to numeric — non-numeric values (DNF, \N, etc.) become NaN
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce")

    # converting Starting Grid to numeric in the same safe way
    df["Starting Grid"] = pd.to_numeric(df["Starting Grid"], errors="coerce")

    # converting Laps to numeric as well
    df["Laps"] = pd.to_numeric(df["Laps"], errors="coerce")

    # converting Points to numeric
    df["Points"] = pd.to_numeric(df["Points"], errors="coerce")

    # creating a boolean DNF flag — True when Position is NaN (could not finish)
    df["DNF"] = df["Position"].isna()

    # computing the places gained: positive = moved forward, negative = fell back
    df["PlacesGained"] = df["Starting Grid"] - df["Position"]

    # flagging whether the driver set the fastest lap for that race
    df["SetFastestLap"] = df["Fastest Lap"].apply(
        lambda x: False if (pd.isna(x) or str(x).strip().lower() in ["", "nan", "none", "\\n", "na"]) else True
    )

    return df


def getCleanData(dataFolder="."):
    """
    running the full load-and-clean pipeline as a single function.
    """
    rawDf = loadAllData(dataFolder)
    cleanDf = cleanData(rawDf)
    return cleanDf


# PHASE 3 — ANALYSIS & VISUALIZATION FUNCTIONS
# Each function below answers one specific analytical question and
# returns a Plotly figure (and sometimes a summary dataframe).


def plotGridAdvantage(df, selectedSeasons, selectedTrack):
    """
    graphing how much the Starting Grid impacts the final Position,
    using a scatter plot to show the variance.
    """
    # filtering down to selected seasons
    filteredDf = df[df["Season"].isin(selectedSeasons)].copy()

    # applying track filter if a specific track was selected
    if selectedTrack != "All Tracks":
        filteredDf = filteredDf[filteredDf["Track"] == selectedTrack]

    # dropping rows where either grid or position is missing
    filteredDf = filteredDf.dropna(subset=["Starting Grid", "Position"])

    fig = px.scatter(
        filteredDf,
        x="Starting Grid",
        y="Position",
        color="Team",
        hover_data=["Driver", "Track", "Season"],
        trendline="ols",
        title="Starting Grid vs. Final Race Position",
        labels=
        {
            "Starting Grid": "Starting Grid Position",
            "Position": "Final Race Position"
        },
        opacity=0.65
    )

    # inverting both axes so that 1st place is at the top-left
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(autorange="reversed")
    fig.update_layout(height=550)

    return fig


def plotTeamReliability(df, selectedSeasons):
    """
    calculating team DNF counts and estimating the points they threw away.
    """
    filteredDf = df[df["Season"].isin(selectedSeasons)].copy()

    # grouping by team to count total DNFs
    dnfDf = filteredDf[filteredDf["DNF"] == True]
    dnfCounts = dnfDf.groupby("Team").size().reset_index(name="DNF Count")

    # estimating lost points: assuming a rough average of 8 points per finish
    dnfCounts["Estimated Points Lost"] = dnfCounts["DNF Count"] * 8

    dnfCounts = dnfCounts.sort_values("DNF Count", ascending=False)

    fig = px.bar(
        dnfCounts,
        x="Team",
        y=["DNF Count", "Estimated Points Lost"],
        barmode="group",
        title="Team Reliability — DNF Count and Estimated Points Lost",
        labels=
        {
            "value": "Count / Points",
            "variable": "Metric"
        },
        color_discrete_sequence=px.colors.qualitative.Bold
    )

    fig.update_layout(xaxis_tickangle=-35, height=520)

    return fig, dnfCounts


def plotFastestLapStrategy(df, selectedSeasons):
    """
    checking if the fastest lap point actually goes to the race winner
    or if midfield drivers are stealing it.
    """
    filteredDf = df[df["Season"].isin(selectedSeasons)].copy()

    # isolating only rows where a driver set the fastest lap
    fastestLapDf = filteredDf[filteredDf["SetFastestLap"] == True].dropna(subset=["Position"])
    
    if fastestLapDf.empty:
        return None, fastestLapDf
        
    # grouping finishing positions into readable buckets
    def bucketPosition(pos):
        if pos == 1:
            return "1st (Winner)"
        elif pos <= 3:
            return "2nd–3rd (Podium)"
        elif pos <= 10:
            return "4th–10th (Points)"
        else:
            return "11th+ (Outside Points)"

    fastestLapDf = fastestLapDf.copy()
    fastestLapDf["Finish Bucket"] = fastestLapDf["Position"].apply(bucketPosition)

    bucketCounts = fastestLapDf["Finish Bucket"].value_counts().reset_index()
    bucketCounts.columns = ["Finish Position Group", "Count"]

    fig = px.pie(
        bucketCounts,
        names="Finish Position Group",
        values="Count",
        title="Fastest Lap Bonus — Who Usually Sets It?",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    fig.update_layout(height=480)

    return fig, fastestLapDf[["Driver", "Team", "Track", "Season", "Position"]].rename(
        columns=
        {
            "Position": "Finishing Position"
        }
    )


def plotOvertakingDifficulty(df, selectedSeasons):
    """
    finding tracks where positions barely change from grid to finish.
    """
    filteredDf = df[df["Season"].isin(selectedSeasons)].copy()
    filteredDf = filteredDf.dropna(subset=["Starting Grid", "Position"])

    # calculating the absolute position change per driver per race
    filteredDf["AbsChange"] = (filteredDf["Starting Grid"] - filteredDf["Position"]).abs()

    # calculating mean absolute change per track — lower number means less overtaking
    trackDifficulty = (
        filteredDf.groupby("Track")["AbsChange"]
        .mean()
        .reset_index()
        .rename(columns={"AbsChange": "Avg Position Change"})
        .sort_values("Avg Position Change")
    )

    fig = px.bar(
        trackDifficulty,
        x="Track",
        y="Avg Position Change",
        title="Track Overtaking Difficulty (Lower = Harder to Overtake)",
        color="Avg Position Change",
        color_continuous_scale="RdYlGn",
        labels=
        {
            "Avg Position Change": "Avg Absolute Position Change"
        }
    )

    fig.update_layout(xaxis_tickangle=-40, height=530, coloraxis_showscale=False)

    return fig, trackDifficulty


def plotBiggestMovers(df, selectedSeasons, topN):
    """
    ranking drivers by their average places gained from starting grid to finish.
    """
    filteredDf = df[df["Season"].isin(selectedSeasons)].copy()
    filteredDf = filteredDf.dropna(subset=["PlacesGained"])

    # calculating average places gained per driver across all their races
    moversDf = (
        filteredDf.groupby("Driver")["PlacesGained"]
        .mean()
        .reset_index()
        .rename(columns={"PlacesGained": "Avg Places Gained"})
        .sort_values("Avg Places Gained", ascending=False)
    )

    # grabbing the top N overachievers and bottom N underachievers
    topMovers = moversDf.head(topN)
    bottomMovers = moversDf.tail(topN)
    combinedMovers = pd.concat([topMovers, bottomMovers]).drop_duplicates()

    # adding a label for the colors
    combinedMovers = combinedMovers.copy()
    combinedMovers["Category"] = combinedMovers["Avg Places Gained"].apply(
        lambda x: "Overachiever" if x >= 0 else "Underachiever"
    )

    combinedMovers = combinedMovers.sort_values("Avg Places Gained", ascending=True)

    fig = px.bar(
        combinedMovers,
        x="Avg Places Gained",
        y="Driver",
        orientation="h",
        color="Category",
        color_discrete_map=
        {
            "Overachiever": "#2ecc71",
            "Underachiever": "#e74c3c"
        },
        title=f"Top & Bottom {topN} Drivers by Average Places Gained",
        labels=
        {
            "Avg Places Gained": "Average Places Gained from Grid"
        }
    )

    fig.update_layout(height=max(400, topN * 45), yaxis_title="")

    return fig


def plotTeammateHeadToHead(df, selectedSeasons, selectedTeam):
    """
    comparing teammates side-by-side to see who is carrying the team.
    """
    filteredDf = df[df["Season"].isin(selectedSeasons)].copy()

    # filtering down to just the team the user picked
    teamDf = filteredDf[filteredDf["Team"] == selectedTeam].copy()

    if teamDf.empty:
        return None, None

    # calculating total points and average finishing position
    h2hDf = teamDf.groupby("Driver").agg(
        totalPoints=("Points", "sum"),
        avgPosition=("Position", "mean"),
        raceCount=("Track", "count")
    ).reset_index()

    h2hDf = h2hDf.sort_values("totalPoints", ascending=False)

    figPoints = px.bar(
        h2hDf,
        x="Driver",
        y="totalPoints",
        color="Driver",
        title=f"Teammate Points Comparison — {selectedTeam}",
        labels=
        {
            "totalPoints": "Total Points"
        }
    )

    figAvg = px.bar(
        h2hDf,
        x="Driver",
        y="avgPosition",
        color="Driver",
        title=f"Teammate Avg Finishing Position — {selectedTeam} (Lower is Better)",
        labels=
        {
            "avgPosition": "Average Finishing Position"
        }
    )

    figAvg.update_yaxes(autorange="reversed")
    figPoints.update_layout(height=400)
    figAvg.update_layout(height=400)

    return figPoints, figAvg


def plotMidfieldBattle(df, selectedSeasons):
    """
    comparing the points spread between the top 3 teams and the midfield (4th-7th).
    """
    filteredDf = df[df["Season"].isin(selectedSeasons)].copy()

    # summing up all points per team per season
    constructorPoints = (
        filteredDf.groupby(["Season", "Team"])["Points"]
        .sum()
        .reset_index()
    )

    # ranking the teams within each season
    constructorPoints["Rank"] = constructorPoints.groupby("Season")["Points"].rank(
        ascending=False, method="min"
    )

    # separating the top 3 from the midfield (4th through 7th)
    top3 = constructorPoints[constructorPoints["Rank"] <= 3].copy()
    top3["Group"] = "Top 3 Teams"

    midfield = constructorPoints[
        (constructorPoints["Rank"] >= 4) & (constructorPoints["Rank"] <= 7)
    ].copy()
    midfield["Group"] = "Midfield (4th–7th)"

    combined = pd.concat([top3, midfield])

    fig = px.box(
        combined,
        x="Group",
        y="Points",
        color="Group",
        points="all",
        hover_data=["Team", "Season"],
        title="Top 3 Teams vs. Midfield — Points Distribution",
        color_discrete_sequence=["#3498db", "#e67e22"]
    )

    fig.update_layout(height=500)

    return fig


def plotConsistencyVsPeak(df, selectedSeasons, topN):
    """
    checking if being a consistent finisher actually scores more points than 
    crashing out half the time but winning the other half.
    """
    filteredDf = df[df["Season"].isin(selectedSeasons)].copy()

    # calculating points, position standard deviation (volatility), and races entered
    driverStats = (
        filteredDf.groupby("Driver")
        .agg(
            totalPoints=("Points", "sum"),
            positionStdDev=("Position", "std"),
            raceCount=("Track", "count")
        )
        .reset_index()
        .dropna(subset=["positionStdDev"])
    )

    # making sure we only look at drivers who actually raced a bit
    driverStats = driverStats[driverStats["raceCount"] >= 5]

    # limiting to the top N drivers so the graph isn't too cluttered
    driverStats = driverStats.sort_values("totalPoints", ascending=False).head(topN)

    fig = px.scatter(
        driverStats,
        x="positionStdDev",
        y="totalPoints",
        text="Driver",
        size="raceCount",
        color="totalPoints",
        color_continuous_scale="Viridis",
        title="Consistency vs. Peak Performance — Std Dev of Position vs. Total Points",
        labels=
        {
            "positionStdDev": "Position Std Dev (Higher = More Chaotic)",
            "totalPoints": "Total Points Scored",
            "raceCount": "Races Entered"
        }
    )

    fig.update_traces(textposition="top center")
    fig.update_layout(height=550, coloraxis_showscale=False)

    return fig, driverStats


def plotSurvivalRate(df, selectedSeasons):
    """
    seeing if just surviving the race and finishing laps correlates with winning championships.
    """
    filteredDf = df[df["Season"].isin(selectedSeasons)].copy()
    filteredDf = filteredDf.dropna(subset=["Laps", "Points"])

    # totaling up the laps and points
    survivalDf = (
        filteredDf.groupby("Driver")
        .agg(
            totalLaps=("Laps", "sum"),
            totalPoints=("Points", "sum"),
            raceCount=("Track", "count")
        )
        .reset_index()
    )

    fig = px.scatter(
        survivalDf,
        x="totalLaps",
        y="totalPoints",
        text="Driver",
        size="raceCount",
        trendline="ols",
        title="Season Survival Rate — Total Laps Completed vs. Total Points",
        labels=
        {
            "totalLaps": "Total Laps Completed",
            "totalPoints": "Total Championship Points",
            "raceCount": "Races Entered"
        },
        color="totalPoints",
        color_continuous_scale="Blues"
    )

    fig.update_traces(textposition="top center")
    fig.update_layout(height=550, coloraxis_showscale=False)

    return fig


def plotPodiumDiversity(df, selectedSeasons):
    """
    calculating if the podiums were shared around or if a few teams completely hogged them.
    """
    filteredDf = df[df["Season"].isin(selectedSeasons)].copy()
    filteredDf = filteredDf.dropna(subset=["Position"])

    # isolating the top 3 finishers
    podiumDf = filteredDf[filteredDf["Position"] <= 3].copy()

    # counting up the trophies
    podiumCounts = podiumDf.groupby("Team").size().reset_index(name="Podium Count")
    podiumCounts = podiumCounts.sort_values("Podium Count", ascending=False)

    # calculating the percentage share
    totalPodiums = podiumCounts["Podium Count"].sum()
    podiumCounts["Share (%)"] = (podiumCounts["Podium Count"] / totalPodiums * 100).round(1)

    fig = px.bar(
        podiumCounts,
        x="Team",
        y="Podium Count",
        color="Share (%)",
        color_continuous_scale="Sunset",
        title="Podium Diversity — How Distributed Were Top-3 Finishes?",
        text="Share (%)",
        labels=
        {
            "Podium Count": "Total Podium Finishes"
        }
    )

    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(xaxis_tickangle=-35, height=520, coloraxis_showscale=False)

    return fig, podiumCounts


# STREAMLIT DASHBOARD — MAIN ENTRY POINT
# Wiring everything together into a multi-page sidebar navigation dashboard.

def main():
    """
    setting up the Streamlit UI and routing the user to their selected page.
    """
    st.set_page_config(
        page_title="F1 Data Dashboard",
        page_icon="🏎️",
        layout="wide"
    )

    # loading and cleaning the data once
    df = getCleanData(".")

    # Sidebar — global controls and navigation menu
    st.sidebar.title("🏎️ F1 Data Dashboard")
    st.sidebar.markdown("---")

    # building season multi-select from what's actually in the data
    availableSeasons = sorted(df["Season"].unique().tolist())
    selectedSeasons = st.sidebar.multiselect(
        "Filter by Season(s)",
        options=availableSeasons,
        default=availableSeasons
    )

    # safeguarding against no season being selected
    if len(selectedSeasons) == 0:
        selectedSeasons = availableSeasons

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Select an Analysis**")

    # defining the navigation menu options
    pages = [
        "The Grid Advantage",
        "Team Reliability",
        "Fastest Lap Strategy",
        "Track Overtaking Difficulty",
        "Biggest Movers",
        "Teammate Head-to-Heads",
        "The Midfield Battle",
        "Consistency vs. Peak Performance",
        "Survival Rate / Laps Completed",
        "Podium Diversity"
    ]

    selectedPage = st.sidebar.radio("", pages)

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Dataset: {len(df)} race entries loaded across {len(availableSeasons)} seasons.")

    # Page routing — rendering the selected analysis

    if selectedPage == "The Grid Advantage":
        st.title("The Grid Advantage")
        st.markdown(
            "How much does a driver's Starting Grid actually impact their final Position? "
            "Does starting on pole position guarantee a win, or is there a lot of variance?"
        )

        allTracks = ["All Tracks"] + sorted(df["Track"].dropna().unique().tolist())
        selectedTrack = st.selectbox("Filter by Track", options=allTracks)

        fig = plotGridAdvantage(df, selectedSeasons, selectedTrack)
        st.plotly_chart(fig, use_container_width=True)

        # calculating correlation for the insight
        filteredForCorr = df[df["Season"].isin(selectedSeasons)].dropna(
            subset=["Starting Grid", "Position"]
        )
        if selectedTrack != "All Tracks":
            filteredForCorr = filteredForCorr[filteredForCorr["Track"] == selectedTrack]

        if len(filteredForCorr) > 2:
            corrVal = filteredForCorr["Starting Grid"].corr(filteredForCorr["Position"])
            st.info(
                f"Ans: The correlation between Starting Grid and Final Position is **{corrVal:.3f}**. "
                f"Since this is close to 1.0, qualifying obviously matters a lot. But looking at the scatter plot, "
                f"there is still a ton of variance. This basically means that even if you start on pole, "
                f"bad pit stops, DNFs, or just getting out-driven can completely ruin your race."
            )

    elif selectedPage == "Team Reliability":
        st.title("Team Reliability")
        st.markdown(
            "Which Team suffered the most retirements (DNFs/null positions)? "
            "Calculate how many potential points teams lost because their cars couldn't finish the race."
        )

        fig, dnfData = plotTeamReliability(df, selectedSeasons)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(dnfData, use_container_width=True)

        if not dnfData.empty:
            worstTeam = dnfData.iloc[0]["Team"]
            worstCount = dnfData.iloc[0]["DNF Count"]
            st.info(
                f"Ans: **{worstTeam}** had the worst luck with {worstCount} DNFs across these seasons. "
                f"That is a ton of potential points thrown away just because the car could not make it to the finish line."
            )

    elif selectedPage == "Fastest Lap Strategy":
        st.title("Fastest Lap Strategy")
        st.markdown(
            "Does the bonus point for the Fastest Lap usually go to the race winner, "
            "or is it usually a driver lower down the field trying to steal an extra point?"
        )

        fig, fastestLapDetail = plotFastestLapStrategy(df, selectedSeasons)
        
        # avoiding a crash if data is empty
        if fastestLapDetail.empty:
            st.warning("No fastest lap data available for the selected seasons. Please choose different seasons.")
        else:
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Fastest Lap Setters — Finishing Positions")
            st.dataframe(fastestLapDetail, use_container_width=True)

            st.info(
                "Ans: Looking at the pie chart, we can see who is actually snatching that extra fastest lap point. "
                "Usually, it is not just the race winner showing off. A lot of times, drivers lower in the points will "
                "pit for fresh soft tires right at the end of the race just to steal the bonus point."
            )
            
    elif selectedPage == "Track Overtaking Difficulty":
        st.title("Track Overtaking Difficulty")
        st.markdown(
            "Are there specific Tracks where it is statistically way harder to overtake? "
            "(Looking for races where the final Position closely matches the Starting Grid.)"
        )

        fig, difficultyData = plotOvertakingDifficulty(df, selectedSeasons)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Average Position Change by Track")
        st.dataframe(difficultyData, use_container_width=True)

        if not difficultyData.empty:
            hardestTrack = difficultyData.iloc[0]["Track"]
            easiestTrack = difficultyData.iloc[-1]["Track"]
            st.info(
                f"Ans: **{hardestTrack}** has the lowest average position changes, meaning it is super hard to overtake there. "
                f"On the flip side, **{easiestTrack}** has crazy position changes, meaning drivers are constantly passing each other all race."
            )

    elif selectedPage == "Biggest Movers":
        st.title("Biggest Movers — Overachievers vs. Underachievers")
        st.markdown(
            "Which Driver gained the most places on average from their Starting Grid, "
            "and who consistently dropped backward?"
        )

        topN = st.slider("Number of drivers to show (top & bottom)", min_value=3, max_value=15, value=8)

        fig = plotBiggestMovers(df, selectedSeasons, topN)
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "Ans: The positive bars are the overachievers—these are the guys who consistently make up places on Sunday, "
            "either because they are great at overtaking or they just had bad qualifying sessions. The negative bars are "
            "the drivers who tend to drop backward after the lights go out."
        )

    elif selectedPage == "Teammate Head-to-Heads":
        st.title("Teammate Head-to-Heads")
        st.markdown(
            "Grouping the data by Team to see which Drivers completely dominated "
            "their teammates in Points and Position."
        )

        availableTeams = sorted(df["Team"].dropna().unique().tolist())
        selectedTeam = st.selectbox("Select a Team", options=availableTeams)

        figPoints, figAvg = plotTeammateHeadToHead(df, selectedSeasons, selectedTeam)

        if figPoints is None:
            st.warning(f"No data available for **{selectedTeam}** in the selected seasons.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(figPoints, use_container_width=True)
            with col2:
                st.plotly_chart(figAvg, use_container_width=True)

            st.info(
                f"Ans: Looking at both charts for **{selectedTeam}**, you can easily see who is carrying the team. "
                f"The driver racking up way more points and a lower average finish is definitely the number one driver."
            )

    elif selectedPage == "The Midfield Battle":
        st.title("The Midfield Battle")
        st.markdown(
            "How close was the point spread for the midfield teams (ranked 4th through 7th) "
            "compared to the top 3 dominant teams?"
        )

        fig = plotMidfieldBattle(df, selectedSeasons)
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "Ans: The box plots show the points spread. You can see the midfield (4th through 7th) usually has a "
            "crazy tight point spread because they are constantly fighting for scraps. Meanwhile, the top 3 teams "
            "usually have a huge gap over the rest of the grid."
        )

    elif selectedPage == "Consistency vs. Peak Performance":
        st.title("Consistency vs. Peak Performance")
        st.markdown(
            "Is it mathematically better for a driver to consistently finish 4th or 5th every race, "
            "or to have a chaotic season with a mix of 1st place wins and complete DNFs?"
        )

        topN = st.slider("Max number of drivers to include", min_value=10, max_value=50, value=25)

        fig, statsData = plotConsistencyVsPeak(df, selectedSeasons, topN)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Driver Consistency Statistics")
        st.dataframe(statsData, use_container_width=True)

        st.info(
            "Ans: The drivers on the left side of the plot are super consistent—they might not win every race, "
            "but they almost always finish in the points. Drivers on the right are super chaotic, bouncing between "
            "podiums and DNFs. Usually, staying consistent and avoiding crashes gets you higher in the championship."
        )

    elif selectedPage == "Survival Rate / Laps Completed":
        st.title("Survival Rate / Laps Completed")
        st.markdown(
            "Is there a strong correlation between the sheer number of Laps a driver "
            "completed over the whole season and their final championship standing?"
        )

        fig = plotSurvivalRate(df, selectedSeasons)
        st.plotly_chart(fig, use_container_width=True)

        # calculating correlation for the insight
        survivalCorr = df[df["Season"].isin(selectedSeasons)].dropna(subset=["Laps", "Points"])
        survivalAgg = survivalCorr.groupby("Driver").agg(
            totalLaps=("Laps", "sum"),
            totalPoints=("Points", "sum")
        ).reset_index()

        if len(survivalAgg) > 2:
            corrVal = survivalAgg["totalLaps"].corr(survivalAgg["totalPoints"])
            st.info(
                f"Ans: The correlation between total laps finished and points is **{corrVal:.3f}**. "
                f"This pretty much proves the old saying: 'to finish first, first you have to finish'. "
                f"If a driver DNF's a lot, it does not matter how fast their car is, they are not winning the championship."
            )

    elif selectedPage == "Podium Diversity":
        st.title("Podium Diversity")
        st.markdown(
            "Did one or two teams completely lock out the top 3 spots all season, "
            "or were the podium finishes highly distributed across different constructors?"
        )

        fig, podiumData = plotPodiumDiversity(df, selectedSeasons)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Podium Finish Breakdown by Team")
        st.dataframe(podiumData, use_container_width=True)

        if not podiumData.empty:
            topTeam = podiumData.iloc[0]["Team"]
            topShare = podiumData.iloc[0]["Share (%)"]
            st.info(
                f"Ans: **{topTeam}** completely hogged the podiums, taking up **{topShare}%** of all top-3 finishes "
                f"in these seasons. The teams at the bottom barely even sniffed a trophy."
            )

# Script entry point
if __name__ == "__main__":
    main()