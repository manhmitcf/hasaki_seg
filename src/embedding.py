
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from enum import Enum
from text_chunker import TextChunker  


class ContentType(Enum):
    """Enum for different types of content chunks"""
    DESCRIPTION = "description"
    GENERAL_INFO = "general_info"
    COMMENT = "comment"
    SPECIFICATION = "specification"
    INGREDIENT = "ingredient"
    GUIDE = "guide"


@dataclass
class ChunkData:
    """Data class for chunk information"""
    text: str
    chunk_id: int
    content_type: ContentType


@dataclass
class ProductEmbedding:
    """Data class for product embedding with metadata"""
    id: str
    values: List[float]
    metadata: Dict[str, Any]



class ProductEmbeddingGenerator:
    """Handles generation of embeddings for different product content types"""
    
    def __init__(self, model: SentenceTransformer, chunker: TextChunker):
        self.model = model
        self.chunker = chunker
        
        # Vietnamese field labels
        self.field_labels = {
            'name': 'Tên sản phẩm',
            'english_name': 'Tên tiếng anh',
            'brand': 'Thương hiệu',
            'category_name': 'Danh mục con',
            'price': 'Giá hiện tại',
            'market_price': 'Giá thị trường',
            'total_rating': 'Tổng số đánh giá',
            'average_rating': 'Đánh giá trung bình',
            'comment': 'Số lượng bình luận',
            'categorys': 'Danh mục chính',
            'item_count_by': 'Số lượng sản phẩm đã mua',
            'data_variant': 'Dữ liệu biến thể',
            'stars': 'Chi tiết đánh giá'
        }
    
    def _create_base_metadata(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Create base metadata for all embeddings"""
        return {
            "product_id": product.get('data_product'),
            "name": product.get('name', ''),
            "english_name": product.get('english_name', ''),
            "categorys": product.get('categorys', ''),
            "category_name": product.get('category_name', ''),
            "brand": product.get('brand', ''),
            "price": product.get('price'),
            "total_rating": product.get('total_rating', ''),
            "average_rating": product.get('average_rating', ''),
            "data_variant": product.get('data_variant', ''),
            "item_count_by": product.get('item_count_by', '')
        }
    
    def _create_product_embedding(self, product_id: str, content_type: ContentType, 
                                text: str, embedding: List[float], 
                                base_metadata: Dict[str, Any], 
                                chunk_id: Optional[int] = None) -> ProductEmbedding:
        """Create a standardized product embedding"""
        embedding_id = product_id + content_type.value
        if chunk_id is not None:
            embedding_id += str(chunk_id)
        
        metadata = base_metadata.copy()
        metadata.update({
            "type": content_type.value,
            "text": text
        })
        
        return ProductEmbedding(
            id=embedding_id,
            values=embedding,
            metadata=metadata
        )
    
    def generate_general_info_embedding(self, product: Dict[str, Any]) -> ProductEmbedding:
        """Generate embedding for general product information"""
        info_parts = []
        
        for field, label in self.field_labels.items():
            value = product.get(field, f'Không có {field.lower()}')
            if field == 'stars':
                value = self.chunker.clean_text(str(value))
            info_parts.append(f"{label}: {value}")
        
        general_info = ". ".join(info_parts) + "."
        cleaned_info = self.chunker.clean_text(general_info)
        embedding = self.model.encode([cleaned_info]).tolist()[0]
        
        base_metadata = self._create_base_metadata(product)
        
        return self._create_product_embedding(
            product.get('data_product'),
            ContentType.GENERAL_INFO,
            cleaned_info,
            embedding,
            base_metadata
        )
    
    def generate_comment_embeddings(self, product: Dict[str, Any]) -> List[ProductEmbedding]:
        """Generate embeddings for product comments"""
        comments = product.get('comments', [])
        if not comments:
            return []
        
        embeddings = []
        base_metadata = self._create_base_metadata(product)
        
        for comment_id, comment in enumerate(comments):
            comment = comment.strip()
            if not comment:
                continue
                
            comment_info = f"Hỏi đáp của khách hàng và Shop: {comment}"
            cleaned_comment = self.chunker.clean_text(comment_info)
            embedding = self.model.encode([cleaned_comment]).tolist()[0]
            
            product_embedding = self._create_product_embedding(
                product.get('data_product'),
                ContentType.COMMENT,
                comment,
                embedding,
                base_metadata,
                comment_id
            )
            embeddings.append(product_embedding)
        
        return embeddings
    
    def generate_specification_embedding(self, product: Dict[str, Any]) -> Optional[ProductEmbedding]:
        """Generate embedding for product specification"""
        specification = product.get('specificationinfo', '')
        if not specification:
            return None
        
        cleaned_spec = self.chunker.clean_text(specification)
        embedding = self.model.encode([cleaned_spec]).tolist()[0]
        base_metadata = self._create_base_metadata(product)
        
        return self._create_product_embedding(
            product.get('data_product'),
            ContentType.SPECIFICATION,
            specification,
            embedding,
            base_metadata
        )
    
    def generate_content_chunk_embeddings(self, product: Dict[str, Any], 
                                        content_field: str, 
                                        content_type: ContentType) -> List[ProductEmbedding]:
        """Generate embeddings for chunked content (description, ingredient, guide)"""
        content = product.get(content_field, '')
        if not content:
            return []
        
        chunks = self.chunker.chunk_text(content)
        embeddings = []
        base_metadata = self._create_base_metadata(product)
        
        # Generate embeddings for all chunks at once for efficiency
        chunk_texts = [chunk.text for chunk in chunks]
        chunk_embeddings = self.model.encode(chunk_texts)
        
        for chunk, embedding in zip(chunks, chunk_embeddings):
            product_embedding = self._create_product_embedding(
                product.get('data_product'),
                content_type,
                chunk.text,
                embedding.tolist(),
                base_metadata,
                chunk.chunk_id
            )
            embeddings.append(product_embedding)
        
        return embeddings
class ProductProcessor:
    """Main class for processing products and generating embeddings"""
    
    def __init__(self, model: SentenceTransformer, chunk_size: int = 300, overlap: int = 50):
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        self.embedding_generator = ProductEmbeddingGenerator(model, self.chunker)
    
    def process_single_product(self, product: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a single product and return all its embeddings"""
        all_embeddings = []
        
        try:
            # General information embedding
            general_embedding = self.embedding_generator.generate_general_info_embedding(product)
            all_embeddings.append(general_embedding.__dict__)
            
            # Description chunks
            desc_embeddings = self.embedding_generator.generate_content_chunk_embeddings(
                product, 'descriptioninfo', ContentType.DESCRIPTION
            )
            all_embeddings.extend([emb.__dict__ for emb in desc_embeddings])
            
            # Comment embeddings
            comment_embeddings = self.embedding_generator.generate_comment_embeddings(product)
            all_embeddings.extend([emb.__dict__ for emb in comment_embeddings])
            
            # Specification embedding
            spec_embedding = self.embedding_generator.generate_specification_embedding(product)
            if spec_embedding:
                all_embeddings.append(spec_embedding.__dict__)
            
            # Ingredient chunks
            ingredient_embeddings = self.embedding_generator.generate_content_chunk_embeddings(
                product, 'ingredientinfo', ContentType.INGREDIENT
            )
            all_embeddings.extend([emb.__dict__ for emb in ingredient_embeddings])
            
            # Guide chunks
            guide_embeddings = self.embedding_generator.generate_content_chunk_embeddings(
                product, 'guideinfo', ContentType.GUIDE
            )
            all_embeddings.extend([emb.__dict__ for emb in guide_embeddings])
            
        except Exception as e:
            print(f"Error processing product {product.get('data_product', 'unknown')}: {str(e)}")
            return []
        
        return all_embeddings
    
    def process_products_batch(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of products with progress tracking"""
        all_documents = []
        
        print(f"Processing {len(products)} products with chunking...")
        
        for i, product in enumerate(products):
            if i % 100 == 0:
                print(f"Processed {i}/{len(products)} products")
            
            product_name = product.get('name', 'No Name')
            print(f"Processing product {i+1}/{len(products)}: {product_name}")
            
            product_embeddings = self.process_single_product(product)
            all_documents.extend(product_embeddings)
            
            print(f"Generated {len(product_embeddings)} embeddings for this product")
            print(f"Total documents so far: {len(all_documents)}")
            
            # Break after first product for testing (remove this in production)
            if i == 0:
                break
        
        print(f"Created {len(all_documents)} document chunks from {len(products)} products")
        return all_documents


def process_products_with_chunking(products: List[Dict[str, Any]], 
                                 model: SentenceTransformer,
                                 chunk_size: int = 300, 
                                 overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Main function to process products with chunking and embedding generation
    
    Args:
        products: List of product dictionaries
        model: SentenceTransformer model for creating embeddings
        chunk_size: Maximum characters per chunk
        overlap: Overlap between chunks
        
    Returns:
        List of processed documents with embeddings
    """
    processor = ProductProcessor(model, chunk_size, overlap)
    return processor.process_products_batch(products)