from pydantic import BaseModel
from typing import List, Optional


class GamesSchema(BaseModel):
    """
    Data model to represent a the endpoint games response.
    """

    id: int
    artworks: Optional[List[int]] = None
    category: Optional[int] = None
    cover: Optional[int] = None
    created_at: int
    external_games: Optional[List[int]] = None
    first_release_date: Optional[int] = None
    game_engines: Optional[List[int]] = None
    game_modes: Optional[List[int]] = None
    genres: Optional[List[int]] = None
    keywords: Optional[List[int]] = None
    name: str
    platforms: Optional[List[int]] = None
    player_perspectives: Optional[List[int]] = None
    release_dates: Optional[List[int]] = None
    screenshots: Optional[List[int]] = None
    similar_games: Optional[List[int]] = None
    slug: Optional[str] = None
    status: Optional[int] = None
    summary: Optional[str] = None
    tags: Optional[List[int]] = None
    themes: Optional[List[int]] = None
    updated_at: int
    url: Optional[str] = None
    videos: Optional[List[int]] = None
    websites: Optional[List[int]] = None
    checksum: Optional[str] = None
    language_supports: Optional[List[int]] = None
    game_status: Optional[int] = None
    game_type: Optional[int] = None


class PlatformSchema(BaseModel):
    """
    Data model to represent a the endpoint platforms response.
    """

    id: int
    category: Optional[int] = None
    created_at: int
    generation: Optional[int] = None
    name: str
    platform_logo: Optional[int] = None
    platform_family: Optional[int] = None
    slug: Optional[str] = None
    updated_at: int
    url: Optional[str] = None
    versions: Optional[List[int]] = None
    websites: Optional[List[int]] = None
    checksum: Optional[str] = None
    platform_type: Optional[int] = None
    alternative_name: Optional[str] = None
    abbreviation: Optional[str] = None
    summary: Optional[str] = None


class PlayerPerspectiveSchema(BaseModel):
    """
    Data model to represent a the endpoint player_perspectives response.
    """

    id: int
    created_at: int
    name: str
    slug: Optional[str] = None
    updated_at: int
    url: Optional[str] = None
    checksum: Optional[str] = None


class GenreSchema(BaseModel):
    """
    Data model to represent a the endpoint genres response.
    """

    id: int
    created_at: int
    name: str
    slug: Optional[str] = None
    updated_at: int
    url: Optional[str] = None
    checksum: Optional[str] = None


class ThemeSchema(BaseModel):
    """
    Data model to represent a the endpoint themes response.
    """

    id: int
    created_at: int
    name: str
    slug: Optional[str] = None
    updated_at: int
    url: Optional[str] = None
    checksum: Optional[str] = None


class CompaniesSchema(BaseModel):
    """
    Data model to represent a the endpoint companies response.
    """

    id: int
    change_date_category: Optional[int] = None
    country: Optional[int] = None
    created_at: int
    description: Optional[str] = None
    developed: Optional[List[int]] = None
    name: str
    slug: Optional[str] = None
    start_date: Optional[int] = None
    start_date_category: Optional[int] = None
    updated_at: int
    url: Optional[str] = None
    websites: Optional[List[int]] = None
    checksum: Optional[str] = None
    status: Optional[int] = None
    logo: Optional[int] = None
    published: Optional[List[int]] = None


class AgeRatingSchema(BaseModel):
    """
    Data model to represent a the endpoint age_ratings response.
    """

    id: int
    category: int
    content_descriptions: Optional[List[int]] = None
    rating: Optional[int] = None
    rating_category: Optional[int] = None
    rating_content_descriptions: Optional[str] = None
    rating_name: Optional[str] = None
    checksum: Optional[str] = None


class LanguageSupportSchema(BaseModel):
    """
    Data model to represent a the endpoint languages response.
    """

    id: int
    created_at: int
    name: str
    native_name: str
    locale: str
    updated_at: int
    url: Optional[str] = None
    checksum: Optional[str] = None


class AgeContentDescriptionSchema(BaseModel):
    """
    Data model to represent a the endpoint age_rating_content_descriptions_v2 response.
    """

    id: int
    description: str
    organization: int
    created_at: int
    updated_at: int
    checksum: Optional[str] = None


class GamesModesSchema(BaseModel):
    """
    Data model to represent a the endpoint game_modes response.
    """

    id: int
    created_at: int
    name: str
    slug: Optional[str] = None
    updated_at: int
    url: Optional[str] = None
    checksum: Optional[str] = None
