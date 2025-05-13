import os
import json
import requests
import base64
from typing import Dict, List, Optional, Any, Type
from flask import Flask, request, jsonify
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_google_genai import GoogleGenerativeAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from flask_cors import CORS
from datetime import datetime
import re

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, origins=["*"])

# Tool for fetching trending YouTube videos
class YouTubeTrendingToolInput(BaseModel):
    query: str = Field(description="Search keyword")
    region_code: str = Field(description="Region code (e.g., IN, US)")
    content_type: str = Field(description="Type of content: shorts, videos, or both")

class YouTubeTrendingTool(BaseTool):
    name: str = "youtube_trending_fetcher"
    description: str = "Fetches top trending YouTube content in a specific query and region"
    args_schema: Type[BaseModel] = YouTubeTrendingToolInput

    def _run(self,query: str, region_code: str, content_type: str) -> Dict[str, Any]:
        api_key = os.getenv("YOUTUBE_API_KEY")
        if not api_key:
            return {"error": "YouTube API key not found"}
        
        url = f"https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "type": "video",
            "maxResults": 4,  # Fetch more to filter by content type
            "regionCode": region_code,
            "q": query,
            "order": "viewCount",
            "publishedAfter": "2025-04-01T00:00:00Z",
            "key": api_key
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            # Filter based on content_type
            filtered_videos = []
            for item in data.get("items", []):
                duration = item.get("contentDetails", {}).get("duration", "")
                # Parse duration
                seconds = 0
                if "PT" in duration:
                    if "M" in duration:
                        minutes_part = duration.split("PT")[1].split("M")[0]
                        seconds += int(minutes_part) * 60
                    if "S" in duration:
                        if "M" in duration:
                            seconds_part = duration.split("M")[1].split("S")[0]
                        else:
                            seconds_part = duration.split("PT")[1].split("S")[0]
                        seconds += int(seconds_part)
                
                # Filter based on content_type
                if (content_type == "shorts" and seconds <= 60) or \
                   (content_type == "videos" and seconds > 60) or \
                   (content_type == "both"):
                    
                    # Get published date and calculate video age
                    published_at = item.get("snippet", {}).get("publishedAt", "")
                    published_date = None
                    video_age_days = None
                    
                    try:
                        if published_at:
                            published_date = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
                            video_age_days = (datetime.now() - published_date).days
                    except Exception:
                        pass
                    
                    # Collect all available tags
                    tags = item.get("snippet", {}).get("tags", [])
                    
                    # Calculate engagement metrics
                    # view_count = int(item.get("statistics", {}).get("viewCount", 0))
                    # like_count = int(item.get("statistics", {}).get("likeCount", 0))
                    # comment_count = int(item.get("statistics", {}).get("commentCount", 0))
                    
                    # like_view_ratio = 0
                    # comment_view_ratio = 0
                    
                    # if view_count > 0:
                    #     like_view_ratio = like_count / view_count
                    #     comment_view_ratio = comment_count / view_count
                    
                    filtered_videos.append({
                        "video_id": item.get("id"),
                        "title": item.get("snippet", {}).get("title"),
                        "description": item.get("snippet", {}).get("description"),
                        "channel_id": item.get("snippet", {}).get("channelId"),
                        "channel_title": item.get("snippet", {}).get("channelTitle"),
                        "published_at": published_at,
                        "video_age_days": video_age_days,
                        "duration_seconds": seconds,
                        "tags": tags,
                        "tag_count": len(tags),
                        # "view_count": view_count,
                        # "like_count": like_count,
                        # "comment_count": comment_count,
                        # "like_view_ratio": like_view_ratio,
                        # "comment_view_ratio": comment_view_ratio,
                    })
                
                if len(filtered_videos) >= 10:
                    break
            
            return {"videos": filtered_videos[:10]}  # Return only top 10 videos
            
        except Exception as e:
            return {"error": str(e)}

# Tool for searching YouTube videos
class YouTubeSearchToolInput(BaseModel):
    query: str = Field(description="Search keyword")
    region_code: str = Field(description="Region code (e.g., IN, US)")
    content_type: str = Field(description="Type of content: shorts, videos, or both")

class YouTubeSearchTool(BaseTool):
    name: str = "youtube_search_fetcher"
    description: str = "Searches for relevant YouTube content based on keyword and category"
    args_schema: Type[BaseModel] = YouTubeSearchToolInput

    def _run(self, query: str, region_code: str, content_type: str) -> Dict[str, Any]:
        api_key = os.getenv("YOUTUBE_API_KEY")
        if not api_key:
            return {"error": "YouTube API key not found"}
        
        url = f"https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "type": "video",
            "maxResults": 4,  # Fetch more to filter by content type
            "regionCode": region_code,
            "q": query,
            "order": "viewCount",
            "publishedAfter": "2025-01-01T00:00:00Z",
            "key": api_key
        }
        
        # Apply duration filter if we're only looking for one type
        if content_type == "shorts":
            params["videoDuration"] = "short"
        elif content_type == "videos":
            params["videoDuration"] = "medium"
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            # print("Search Data:",data)
            
            # Extract video IDs for additional data fetch
            video_ids = []
            for item in data.get("items", []):
                video_id = item.get("id", {}).get("videoId")
                if video_id:
                    video_ids.append(video_id)
            
            # Get detailed video information
            if video_ids:
                videos_url = f"https://www.googleapis.com/youtube/v3/videos"
                videos_params = {
                    "part": "snippet,contentDetails,statistics",
                    "id": ",".join(video_ids),
                    "key": api_key
                }
                
                videos_response = requests.get(videos_url, videos_params)
                videos_data = videos_response.json()
                
                # Filter and process videos
                filtered_videos = []
                for item in videos_data.get("items", []):
                    duration = item.get("contentDetails", {}).get("duration", "")
                    # Parse duration
                    seconds = 0
                    if "PT" in duration:
                        if "M" in duration:
                            minutes_part = duration.split("PT")[1].split("M")[0]
                            seconds += int(minutes_part) * 60
                        if "S" in duration:
                            if "M" in duration:
                                seconds_part = duration.split("M")[1].split("S")[0]
                            else:
                                seconds_part = duration.split("PT")[1].split("S")[0]
                            seconds += int(seconds_part)
                    
                    # Filter based on content_type
                    if (content_type == "shorts" and seconds <= 60) or \
                       (content_type == "videos" and seconds > 60) or \
                       (content_type == "both"):
                        
                        # Get published date and calculate video age
                        published_at = item.get("snippet", {}).get("publishedAt", "")
                        published_date = None
                        video_age_days = None
                        
                        try:
                            if published_at:
                                published_date = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
                                video_age_days = (datetime.now() - published_date).days
                        except Exception:
                            pass
                        
                        # Collect all available tags
                        tags = item.get("snippet", {}).get("tags", [])
                        
                        # Calculate engagement metrics
                        # view_count = int(item.get("statistics", {}).get("viewCount", 0))
                        # like_count = int(item.get("statistics", {}).get("likeCount", 0))
                        # comment_count = int(item.get("statistics", {}).get("commentCount", 0))
                        
                        # like_view_ratio = 0
                        # comment_view_ratio = 0
                        
                        # if view_count > 0:
                        #     like_view_ratio = like_count / view_count
                        #     comment_view_ratio = comment_count / view_count
                        
                        filtered_videos.append({
                            "video_id": item.get("id"),
                            "title": item.get("snippet", {}).get("title"),
                            "description": item.get("snippet", {}).get("description"),
                            "channel_id": item.get("snippet", {}).get("channelId"),
                            "channel_title": item.get("snippet", {}).get("channelTitle"),
                            "published_at": published_at,
                            "video_age_days": video_age_days,
                            "duration_seconds": seconds,
                            "tags": tags,
                            "tag_count": len(tags),
                            # "view_count": view_count,
                            # "like_count": like_count,
                            # "comment_count": comment_count,
                            # "like_view_ratio": like_view_ratio,
                            # "comment_view_ratio": comment_view_ratio,
                        })
                    
                    if len(filtered_videos) >= 10:
                        break
                
                return {"videos": filtered_videos[:10]}  # Return only top 10 results
            else:
                return {"videos": []}
            
        except Exception as e:
            return {"error": str(e)}

# Tool for deep video analysis
class VideoAnalysisToolInput(BaseModel):
    video_ids: List[str] = Field(description="List of YouTube video ID")
    content_type: str = Field(description="Type of content: shorts, videos, or both")

class VideoAnalysisTool(BaseTool):
    name: str = "video_content_analyzer"
    description: str = "Performs deep analysis of YouTube videos' content, metadata, and audience engagement"
    args_schema: Type[BaseModel] = VideoAnalysisToolInput  # This should now accept List[str]

    def _run(self, video_ids: List[str], content_type: str) -> List[Dict[str, Any]]:
        api_key = os.getenv("YOUTUBE_API_KEY")
        if not api_key:
            return [{"error": "YouTube API key not found"}]

        video_id_str = ",".join(video_ids[:50])  # Max 50 IDs per request
        url = "https://www.googleapis.com/youtube/v3/videos"
        params = {
            "part": "snippet,contentDetails,statistics,topicDetails",
            "id": video_id_str,
            "key": api_key
        }

        try:
            response = requests.get(url, params=params)
            video_data = response.json()

            if not video_data.get("items"):
                return [{"error": "No videos found"}]

            results = []

            for video_item in video_data["items"]:
                try:
                    video_id = video_item["id"]
                    duration = video_item.get("contentDetails", {}).get("duration", "")
                    
                    # Duration parsing
                    seconds = 0
                    if "PT" in duration:
                        if "M" in duration:
                            minutes_part = duration.split("PT")[1].split("M")[0]
                            seconds += int(minutes_part) * 60
                        if "S" in duration:
                            if "M" in duration:
                                seconds_part = duration.split("M")[1].split("S")[0]
                            else:
                                seconds_part = duration.split("PT")[1].split("S")[0]
                            seconds += int(seconds_part)

                    # Type check
                    if (content_type == "shorts" and seconds > 60) or \
                       (content_type == "videos" and seconds <= 60):
                        results.append({"video_id": video_id, "error": f"Does not match content type '{content_type}'"})
                        continue

                    # Fetch comments
                    comments_url = f"https://www.googleapis.com/youtube/v3/commentThreads"
                    comments_params = {
                        "part": "snippet",
                        "videoId": video_id,
                        "maxResults": 100,
                        "order": "relevance",
                        "key": api_key
                    }

                    try:
                        comments_response = requests.get(comments_url, params=comments_params)
                        comments_data = comments_response.json()
                        comments = []
                        for item in comments_data.get("items", [])[:3]:
                            snippet = item["snippet"]["topLevelComment"]["snippet"]
                            comments.append({
                                "text": snippet.get("textDisplay", ""),
                                "like_count": snippet.get("likeCount", 0),
                                "published_at": snippet.get("publishedAt", "")
                            })
                    except:
                        comments = []

                    # Fetch channel info
                    channel_id = video_item.get("snippet", {}).get("channelId")
                    channel_url = f"https://www.googleapis.com/youtube/v3/channels"
                    channel_params = {
                        "part": "snippet,statistics,brandingSettings",
                        "id": channel_id,
                        "key": api_key
                    }
                    try:
                        channel_response = requests.get(channel_url, params=channel_params)
                        channel_data = channel_response.json()
                        channel_info = channel_data.get("items", [{}])[0]
                    except:
                        channel_info = {}

                    # LLM analysis
                    try:
                        gemini_api_key = os.getenv("GEMINI_API_KEY")
                        model = ChatGoogleGenerativeAI(
                            model="gemini-2.0-flash",
                            google_api_key=gemini_api_key,
                            temperature=0
                        )
                        video_url = f"https://www.youtube.com/watch?v={video_id}"
                        messages = [
                            SystemMessage(content="You are an expert video content analyzer."),
                            HumanMessage(content=f"Analyze this YouTube {'short' if seconds <= 60 else 'video'}: {video_url}...")
                        ]
                        response = model.invoke(messages)
                        video_analysis = response.content
                    except Exception as e:
                        video_analysis = f"Error analyzing video content: {str(e)}"

                    # Metrics calculation
                    published_at = video_item.get("snippet", {}).get("publishedAt", "")
                    try:
                        published_date = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
                        video_age_days = (datetime.now() - published_date).days
                    except:
                        video_age_days = None

                    view_count = int(video_item.get("statistics", {}).get("viewCount", 0))
                    like_count = int(video_item.get("statistics", {}).get("likeCount", 0))
                    comment_count = int(video_item.get("statistics", {}).get("commentCount", 0))

                    views_per_day = view_count / video_age_days if video_age_days and video_age_days > 0 else 0
                    likes_per_day = like_count / video_age_days if video_age_days and video_age_days > 0 else 0
                    comments_per_day = comment_count / video_age_days if video_age_days and video_age_days > 0 else 0

                    like_view_ratio = like_count / view_count if view_count > 0 else 0
                    comment_view_ratio = comment_count / view_count if view_count > 0 else 0
                    engagement_rate = (like_count + comment_count) / view_count if view_count > 0 else 0

                    results.append({
                        "video_id": video_id,
                        "metadata": {
                            "title": video_item.get("snippet", {}).get("title"),
                            "description": video_item.get("snippet", {}).get("description"),
                            "tags": video_item.get("snippet", {}).get("tags", []),
                            "tag_count": len(video_item.get("snippet", {}).get("tags", [])),
                            "publishedAt": published_at,
                            "video_age_days": video_age_days,
                            "categoryId": video_item.get("snippet", {}).get("categoryId"),
                            "duration_seconds": seconds,
                            "duration_formatted": duration
                        },
                        "statistics": {
                            "viewCount": view_count,
                            "likeCount": like_count,
                            "commentCount": comment_count,
                            "views_per_day": views_per_day,
                            "likes_per_day": likes_per_day,
                            "comments_per_day": comments_per_day,
                            "like_view_ratio": like_view_ratio,
                            "comment_view_ratio": comment_view_ratio,
                            "engagement_rate": engagement_rate
                        },
                        "channel": {
                            "id": channel_id,
                            "title": channel_info.get("snippet", {}).get("title"),
                            "description": channel_info.get("snippet", {}).get("description"),
                            "subscriberCount": channel_info.get("statistics", {}).get("subscriberCount"),
                            "videoCount": channel_info.get("statistics", {}).get("videoCount"),
                            "country": channel_info.get("snippet", {}).get("country")
                        },
                        "comments": comments,
                        "content_analysis": video_analysis,
                        "video_url": video_url
                    })
                except Exception as e:
                    results.append({"video_id": video_item.get("id"), "error": str(e)})

            return results

        except Exception as e:
            return [{"error": str(e)}]

# CrewAI setup
class YouTubeContentCrew:
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        self.llm = GoogleGenerativeAI(
            model="gemini/gemini-1.5-flash",
            google_api_key=self.gemini_api_key,
            temperature=0.5,
        )
        
        self.trending_tool = YouTubeTrendingTool()
        self.search_tool = YouTubeSearchTool()
        self.analysis_tool = VideoAnalysisTool()
        
        self._setup_agents()
        self._setup_crew()
    
    def _setup_agents(self):
        self.category_topic_decider = Agent(
            role="Category & Topic Decider",
            goal="Determine the most relevant trending videos with query and region based on user prompt",
            backstory="You are an expert in YouTube content trends and categorization. You have deep knowledge of what makes content perform well across different regions and categories.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.trending_tool]
        )
        
        self.keyword_search_decider = Agent(
            role="Keyword & Search Decider",
            goal="Determine the most effective search keywords and parameters to find relevant content",
            backstory="You are a search optimization specialist with extensive knowledge of YouTube's search algorithms and trends. You excel at crafting precise search queries that yield the most relevant content.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.search_tool]
        )
        
        self.video_selector = Agent(
            role="Video Selection Expert",
            goal="Select the most relevant and high-potential videos from trending and search results and also do analysis for remaining videos also",
            backstory="You are a content curation expert with a keen eye for what makes videos successful. You can identify the most promising content that balances viral appeal with niche-specific value.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        self.content_analyzer = Agent(
            role="Deep Video Content Analyzer",
            goal="Perform comprehensive analysis of selected videos to extract actionable insights and Gather information about remaining videos",
            backstory="You are a video content analysis expert with experience in deconstructing successful videos. You can identify the technical, creative, and strategic elements that drive engagement.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.analysis_tool]
        )
        
        self.marketing_strategist = Agent(
            role="Marketing Strategy Expert",
            goal="Develop actionable marketing strategies based on video analysis",
            backstory="You are a digital marketing strategist specializing in video content. You excel at translating video analysis into practical, actionable marketing advice for content creators.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    def _setup_crew(self):
        self.crew = Crew(
            agents=[
                self.category_topic_decider,
                self.keyword_search_decider,
                self.video_selector,
                self.content_analyzer,
                self.marketing_strategist
            ],
            tasks=[],
            verbose=True,
            process=Process.sequential
        )
    
    def _create_tasks(self, user_prompt: str, content_type: str, region_code: str):
        # Task 1: Determine categories, topics and fetch trending videos
        trending_task = Task(
            description=f"""Based on the user prompt: "{user_prompt}", determine:
            
            1. Pass the exact keyword passed by user if its a keyword, else if it is like a prompt just use the main one single worded keyword.
            2. Use the provided region code: {region_code} (or default to IN if invalid)
            3. Content type is: {content_type} (shorts, videos, or both)
            4. Check the parameters to pass correctly and use the tool perfectly (Don't use the category ID here)
            
            Once you've determined these parameters, use the youtube_trending_fetcher tool to fetch the top 10 trending videos in that category and region that match the content type.
            
            IMPORTANT: Your output MUST include your reasoning for selecting the category, followed by the complete results from the trending fetcher tool.
            """,
            expected_output="Analysis of the user prompt with category and region decisions plus trending video results",
            agent=self.category_topic_decider,
            tools=[self.trending_tool]
        )
        
        # Task 2: Determine search parameters and fetch search videos
        search_task = Task(
            description=f"""Based on the user prompt: "{user_prompt}", determine:
            
            1. Pass the exact keyword passed by user if its a keyword, else if it is like a prompt just use the main one single worded keyword.
            2. Use the provided region code: {region_code} (or default to IN if invalid)
            3. Content type is: {content_type} (shorts, videos, or both)
            4. Check the parameters to pass correctly and use the tool perfectly (Don't use the category ID here)
            
            Once you've determined these parameters, use the youtube_search_fetcher tool to search for the top 10 relevant videos that match the content type.
            
            IMPORTANT: Your output MUST include your reasoning for selecting the category, search keyword, and date range, followed by the complete results from the search fetcher tool.
            """,
            expected_output="Analysis of the user prompt with category, search keyword, region, and date decisions plus search video results",
            agent=self.keyword_search_decider,
            tools=[self.search_tool]
        )
        
        # Task 3: Select the best videos from trending and search results
        selection_task = Task(
            description="""Review both the trending videos and search videos obtained in the previous tasks.
            
            From these two sets (10 trending videos + 10 search videos), select:
            1. The single BEST trending video (high domain relevance, general appeal)
            2. The single BEST search video (more niche, insightful but not as viral)
            
            For each selected video, explain:
            - Why you selected it over the others
            - What specific elements make it the best choice
            - How it relates to the user's original prompt
            
            ALSO PASS THE REMAINING VIDEOS TOO AS OTHER SIMILAR VIDEOS
            
            IMPORTANT: Your output MUST include detailed justification for each selection. For the selection, only include video IDs, titles, and descriptions (not full metadata). Store all other metadata for later use.
            """,
            expected_output="Selection of the best trending video and best search video with detailed justification and also the remaining videos as other similar videos",
            agent=self.video_selector,
            context=[trending_task, search_task]
        )
        
        # Task 4: Perform deep analysis on the selected videos
        analysis_task = Task(
            description=f"""For each of the two selected videos, use the video_content_analyzer tool to perform a comprehensive analysis.
            
            Make sure to:
            
            PASSING INPUT: You can now pass all the video ids as a array and get the result with the tool.
            
            1. Use the content_type: {content_type} parameter when analyzing the videos
            2. Analyze all available metadata, statistics, and content
            3. Extract insights about visual elements, audio, editing style, storytelling approach of the top selected videos
            4. Identify patterns in audience engagement (comments, likes, etc.)
            5. Analyze current trends and future trend predictions
            6. Do deep video analysis for top 2 videos and for the remaining videos get only the snippet, statistics and contentData for view, likes, comments data.
            
            IMPORTANT: For the top 2 video, compile a detailed analysis report that includes key findings about what makes the content successful for remaining just metrics data is enough
            
            NOTE: The difference is for top 2 videos you do metadata and the in video deep analysis where as for remaining videos just metadata (numbers) is needed.
            """,
            expected_output="Comprehensive analysis reports for the two selected videos and also the metadata information of remaining videos too",
            agent=self.content_analyzer,
            tools=[self.analysis_tool],
            context=[selection_task]
        )
        
        # Task 5: Develop marketing strategy based on all insights
        strategy_task = Task(
            description=f"""Based on the original user prompt: "{user_prompt}" and all the analysis performed in previous tasks, develop a comprehensive marketing strategy.
            
            Your strategy should include:
            
            IMPORTANT: (Don't be generic be unique, attention seeking, knowledgable enough based on the video analysis report)
            
            1. Content Recommendations 
               - Specific content types to create
               - Visual style recommendations
               - Audio/music recommendations
               - Storytelling approach
               - Editing style and pacing
            
            2. Marketing Tactics:
               - Recommended tags and keywords
               - Title and description optimization
               - Thumbnail design recommendations
               - Best posting times and frequency
               - Audience engagement strategies
            
            3. Success Metrics:
               - How to measure effectiveness
               - Expected engagement patterns
               - Growth opportunities
               - Count for each keyword among all 10 videos (Top 10 keywords)
               
            4. Video Organization:
               - Analyzed Videos: The 2 deeply analyzed videos
               - Top Matches: The remaining top 3 videos from trending and top 3 from search
               - Similar Content: 5 videos similar to the analyzed ones
               - Trending Content: 5 additional trending videos
               
            5. Trend Analysis:
               - Current trends identified
               - Future trend predictions
               
            IMPORTANT NOTE: STRICTLY GENERATE ALL THE NUMERICALS IN INT EVEN THOUGH I HAVE USED QOUTES
            
            IMPORTANT: Your output should be in proper JSON format that the user can immediately use.
            Please respond ONLY in valid, parseable JSON format, no explanations or extra text. Ensure the JSON is well-formed and passes JSON linting.
            Whatever error happens, any tool malfunctions also don't give any response other than the JSON Data
            """,
            expected_output="""{
                "marketing_strategy": {
                    "target_audience": "<Describe your target audience here>",
                    "overall_goal": "<State your overall marketing goal here>",
                    "content_recommendations": {
                        "content_types": [
                            "<Type of video/content 1>",
                            "<Type of video/content 2>",
                            "<Type of video/content 3>"
                        ],
                        "visual_style": "<Describe your desired visual style here>",
                        "audio_music": "<Describe your audio/music requirements>",
                        "storytelling_approach": "<Explain your storytelling style and tone>",
                        "editing_style_and_pacing": "<Describe editing style, pacing, and duration goals>"
                    },
                    "marketing_tactics": {
                        "recommended_tags_and_keywords": [
                            ["<keyword1>",count],
                            ["<keyword2>",count],
                            ["<keyword3>",count]
                        ],
                        "title_and_description_optimization": "<Best practices for titles and descriptions>",
                        "thumbnail_design_recommendations": "<Tips for thumbnail creation>",
                        "best_posting_times_and_frequency": "<Your posting schedule recommendations>",
                        "audience_engagement_strategies": "<How to drive engagement and foster community>"
                    },
                    "success_metrics": {
                        "how_to_measure_effectiveness": "<Define your key performance indicators>",
                        "expected_engagement_patterns": "<Expected engagement trends>",
                        "growth_opportunities": "<Ideas for expanding reach and impact>"
                    },
                    "trend_analysis": {
                        "current_trends": "<Current content trends identified in the analysis>",
                        "future_predictions": "<Predictions about where these trends are heading>"
                    },
                    "videos": {
                        "analyzed_videos": [
                            {
                                "video_id": "<Video ID of analyzed trending video>",
                                "title": "<Video title>",
                                "description": "<Video description>",
                                "statistics": {
                                    "views": <Number of views> STRICTLY INT,
                                    "likes": <Number of likes> STRICTLY INT,
                                    "comments": <Number of comments> STRICTLY INT,
                                    "subscribers": <Channel subscriber count> STRICTLY INT,
                                    "views_per_day": <Average views per day> STRICTLY INT,
                                    "engagement_rate": <Engagement rate>  STRICTLY INT
                                },
                                "analysis": "<Key insights from deep content analysis>",
                                "current_trends": "<Current trends this video follows>",
                                "future_trends": "<Future trend predictions>",
                                "video_url": "<Video URL>"
                            },
                            {
                                "video_id": "<Video ID of analyzed search video>",
                                "title": "<Video title>",
                                "description": "<Video description>",
                                "statistics": {
                                    "views": <Number of views> STRICTLY INT,
                                    "likes": <Number of likes> STRICTLY INT,
                                    "comments": <Number of comments> STRICTLY INT,
                                    "subscribers": <Channel subscriber count> STRICTLY INT,
                                    "views_per_day": <Average views per day> STRICTLY INT,
                                    "engagement_rate": <Engagement rate>  STRICTLY INT
                                },
                                "analysis": "<Key insights from deep content analysis>",
                                "current_trends": "<Current trends this video follows>",
                                "future_trends": "<Future trend predictions>",
                                "video_url": "<Video URL>"
                            }
                        ],
                        "top_matches": {
                            "trending": [
                                {
                                    "video_id": "<Video ID>",
                                    "title": "<Video title>",
                                    "statistics": {
                                        "views": <Number of views> STRICTLY INT,
                                        "likes": <Number of likes> STRICTLY INT,
                                        "comments": <Number of comments> STRICTLY INT,
                                    },
                                    "video_url": "<Video URL>"
                                }
                            ],
                            "search": [
                                {
                                    "video_id": "<Video ID>",
                                    "title": "<Video title>",
                                    "statistics": {
                                        "views": <Number of views> STRICTLY INT,
                                        "likes": <Number of likes> STRICTLY INT,
                                        "comments": <Number of comments> STRICTLY INT,
                                    },
                                    "video_url": "<Video URL>"
                                }
                            ]
                        },
                        "similar_content": [
                            {
                                "video_id": "<Video ID>",
                                "title": "<Video title>",
                                "statistics": {
                                    "views": <Number of views> STRICTLY INT,
                                    "likes": <Number of likes> STRICTLY INT,
                                    "comments": <Number of comments> STRICTLY INT,
                                },
                                "video_url": "<Video URL>"
                            }
                        ],
                        "trending_content": [
                            {
                                "video_id": "<Video ID>",
                                "title": "<Video title>",
                                "statistics": {
                                     "views": <Number of views> STRICTLY INT,
                                     "likes": <Number of likes> STRICTLY INT,
                                     "comments": <Number of comments> STRICTLY INT,
                                },
                                "video_url": "<Video URL>"
                            }
                        ]
                    }
                }
            }
            Please respond ONLY in valid, parseable JSON format, no explanations or extra text. Ensure the JSON is well-formed and passes JSON linting.
            """,
            agent=self.marketing_strategist,
            context=[trending_task, search_task, selection_task, analysis_task]
        )
        
        return [trending_task, search_task, selection_task, analysis_task, strategy_task]
    
    def analyze_prompt(self, user_prompt: str, content_type: str, region_code: str) -> Dict[str, Any]:
        tasks = self._create_tasks(user_prompt, content_type, region_code)
        self.crew.tasks = tasks
        result = self.crew.kickoff()
        
        # Extract outputs from each task
        outputs = {}
        try:
            for task in tasks:
                if task.agent.role == "Marketing Strategy Expert":
                    content = str(task.output)
                    
                    if content.startswith("```json"):
                        content = re.sub(r"```json\s*", "", content)
                        content = re.sub(r"```\s*$", "", content)
            
            content = json.loads(content)
                    
        except Exception as e:
            print(f"Error extracting results: {str(e)}")
        
        return content

# Main Flask route
@app.route('/analyze-shorts', methods=['POST'])
def analyze_shorts():
    try:
        data = request.json
        user_prompt = data.get('prompt', '')
        content_type = data.get('content_type', '')
        region_code = data.get('region_code', '')
        
        if not user_prompt:
            return jsonify({
                'status': 'error',
                'message': 'User prompt is required'
            }), 400
        
        # Initialize YouTubeShortsCrew and analyze the prompt
        shorts_analyzer = YouTubeContentCrew()
        result = shorts_analyzer.analyze_prompt(user_prompt,content_type,region_code)
        
        return jsonify({
            'status': 'success',
            'data': result
        })
    
    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=10000)