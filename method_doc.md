# HSM: Hierarchical Scene Motifs - Complete Method Documentation

## 1. Research Method Overview

### 1.1 Core Concept
HSM (Hierarchical Scene Motifs) is a novel framework for generating dense, realistic 3D indoor scenes from natural language descriptions. The key innovation is treating indoor scenes as hierarchical compositions of recurring spatial patterns (motifs) across multiple scales, rather than randomly placing individual objects.

### 1.2 Key Innovations
- **Multi-Scale Hierarchy**: Processes objects at 4 distinct scales: Large (furniture), Wall (mounted objects), Ceiling (fixtures), and Small (surface objects)
- **Scene Motifs**: Introduces reusable spatial patterns that capture common object arrangements (e.g., "seating area", "dining setup")
- **Unified Generation**: Single framework handles all object scales, unlike prior work focusing only on large furniture
- **Cross-Scale Dependencies**: Objects at finer scales depend on coarser scales (e.g., small objects placed on tables generated earlier)

### 1.3 Problem Addressed
Previous methods either:
- Focus only on large furniture, leaving scenes empty
- Place small objects randomly without semantic coherence
- Cannot follow detailed text descriptions accurately
- Lack understanding of functional object groupings

## 2. Technical Architecture

### 2.1 Hierarchical Processing Pipeline

The system processes scenes in a strict hierarchical order:

```
1. Room Layout Generation (if not provided)
   ↓
2. Large Object Processing (floor-supported furniture)
   ↓
3. Wall Object Processing (wall-mounted items)
   ↓
4. Ceiling Object Processing (ceiling fixtures)
   ↓
5. Small Object Processing (surface-placed items)
```

Each stage builds upon previous stages, creating support regions for subsequent object placement.

### 2.2 Core Components

#### 2.2.1 Scene Representation
```python
class Scene:
    - room_polygon: Shapely Polygon defining floor boundary
    - room_height: Float for ceiling height
    - door_location: Optional door position
    - window_locations: List of window positions
    - motifs: Dict[ObjectType, List[SceneMotif]]
    - scene_spec: SceneSpec with object specifications
```

#### 2.2.2 Scene Motif Structure
```python
class SceneMotif:
    - id: Unique identifier
    - arrangement: Spatial arrangement of objects
    - object_specs: List of furniture specifications
    - position: (x, y, z) world coordinates
    - rotation: Degrees around Y-axis
    - extents: (width, height, depth) bounding box
    - wall_alignment: Boolean for wall constraint
    - ignore_collision: Boolean for collision override
```

#### 2.2.3 Arrangement Representation
```python
class Arrangement:
    - objs: List[Obj] with bounding boxes
    - description: Natural language description
    - program_str: Executable spatial program
    - glb_path: Path to 3D model file
```

### 2.3 Scene Motif System

#### 2.3.1 Motif Types (from motif_definitions.yaml)

**Single Object Motifs:**
- `stack`: Vertical stacking with uniform spacing
- `pile`: Randomized pile configuration
- `row`: Horizontal line with configurable spacing
- `grid`: 2D grid pattern
- `pyramid`: Pyramid shape with decreasing layers
- `wall_grid`: Grid pattern on walls
- `wall_vertical_column`: Vertical column on walls
- `wall_horizontal_row`: Horizontal row on walls

**Two Object Motifs:**
- `face_to_face`: Objects facing each other
- `bed_setup`: Bed with flanking nightstands
- `surround`: Objects around a central piece
- `wall_gallery`: Gallery arrangement on walls

**Multi-Object Motifs:**
- `dining_set`: Table with surrounding chairs
- `living_area`: Sofa, coffee table, TV setup
- `workspace`: Desk with chair and accessories
- `vanity_setup`: Vanity with mirror and stool

#### 2.3.2 Motif Decomposition Process

1. **LLM Analysis**: GPT-4 analyzes object list and description to identify appropriate motifs
2. **Hierarchical Decomposition**: Complex arrangements decomposed into:
   - Primary arrangement (main motif)
   - Secondary arrangements (supporting motifs)
   - Compositional arrangement (combining sub-motifs)
3. **Program Synthesis**: Each motif converted to executable spatial program

### 2.4 Spatial Program Language

The system uses a custom DSL for spatial relationships:

```python
# Basic operations
create(object_type, dimensions)
move(object, x, y, z)
rotate(object, axis, degrees)

# Spatial relations
face_to_face(obj1, obj2, distance)
surround(center_obj, surrounding_objs, radius)
stack(objects, spacing)
align_to_wall(object, wall_id)

# Meta-programs for complex arrangements
dining_set(table, chairs) -> Arrangement
workspace(desk, chair, accessories) -> Arrangement
```

### 2.5 Support Region Analysis

#### 2.5.1 Support Surface Extraction
The system analyzes 3D meshes to identify valid support surfaces:

```python
def analyze_support_surfaces(mesh):
    # Extract horizontal surfaces (normal.y > 0.9)
    # Filter by area threshold (> 0.01 m²)
    # Group by height layers
    # Calculate available space above each layer
    # Return dict of support regions with properties
```

Key parameters:
- `NORMAL_HORIZONTAL_THRESHOLD = 0.9`: Surface horizontality check
- `MIN_AREA = 0.01`: Minimum surface area in m²
- `MIN_AVAILABLE_SPACE_ABOVE_LAYER = 0.3`: Minimum clearance in meters
- `MERGE_DISTANCE = 0.05`: Distance for merging adjacent surfaces

#### 2.5.2 Surface Hierarchy
- **Primary surfaces**: Tables, desks, counters (0.6-1.0m height)
- **Secondary surfaces**: Shelves, cabinets (variable height)
- **Floor surfaces**: Direct floor placement
- **Wall surfaces**: Wall-mounted regions

## 3. Implementation Details

### 3.1 LLM Integration

#### 3.1.1 Session Management
```python
class Session:
    - model: "gpt-4o-2024-08-06" (default)
    - temperature: 0.7
    - retry_count: 5
    - validation_functions: Dict[str, Callable]
    - past_messages: Conversation history
```

#### 3.1.2 Prompt Engineering Strategy
Each stage has specialized prompts:
- `scene_prompts_room.yaml`: Room layout generation
- `scene_prompts_large.yaml`: Large furniture arrangement
- `scene_prompts_wall.yaml`: Wall object placement
- `scene_prompts_ceiling.yaml`: Ceiling fixture placement
- `scene_prompts_small.yaml`: Small object population
- `sm_prompts_decompose.yaml`: Motif decomposition
- `sm_prompts_inference.yaml`: Motif inference

#### 3.1.3 Validation and Retry Mechanism
```python
def send_with_validation(task, data, validation_func, retry=5):
    for attempt in range(retry):
        response = send_to_llm(task, data)
        valid, error_msg, error_idx = validation_func(response)
        if valid:
            return response
        # Provide specific feedback for retry
        add_feedback(error_msg)
```

Validation functions check:
- JSON structure validity
- Object ID consistency
- Dimension constraints
- Spatial relationship feasibility
- Room boundary compliance

### 3.2 CLIP-Based Retrieval System

#### 3.2.1 Embedding Infrastructure
```python
# Precomputed embeddings for HSSD dataset
embeddings: torch.Tensor  # Shape: [num_objects, 512]
index: List[str]  # Object IDs mapping

# Similarity computation
def compute_similarities(text_queries, embeddings):
    text_features = clip_model.encode_text(text_queries)
    similarities = cosine_similarity(text_features, embeddings)
    return top_k_indices
```

#### 3.2.2 Retrieval Pipeline
1. **Text Enhancement**: Augment object names with contextual descriptions
2. **WordNet Filtering**: Use synset keys to pre-filter candidates
3. **CLIP Ranking**: Compute semantic similarities
4. **Dimension Filtering**: Match size constraints
5. **Support Surface Matching**: Ensure compatibility with target surfaces
6. **Deduplication**: Avoid reusing same models

Key parameters:
- `top_k = 10`: Number of candidates to consider
- `similarity_threshold = 0.15`: Minimum semantic similarity
- `size_tolerance = 0.3`: Dimension matching tolerance

### 3.3 Spatial Optimization

#### 3.3.1 DFS Solver
```python
class DFSSolver:
    config:
        grid_size: 0.1  # Discretization in meters
        max_duration: 10.0  # Timeout in seconds
        max_candidates_per_motif: 10
        alignment_threshold: -0.7  # Dot product for alignment
        
    constraints:
        - Collision avoidance (unless ignore_collision)
        - Wall alignment (if specified)
        - Support validation
        - Room boundary containment
```

#### 3.3.2 Optimization Stages

**Stage 1: Motif-Level Optimization**
```python
def optimize_motif(arrangement):
    # Resolve internal collisions
    # Apply gravity approximation
    # Tighten object spacing
    # Maintain relative positions
```

**Stage 2: Scene-Level Optimization**
```python
def optimize_scene(motifs):
    # Place motifs using DFS solver
    # Resolve inter-motif collisions
    # Validate support relationships
    # Apply wall constraints
```

**Stage 3: Hierarchical Optimization**
```python
def optimize_hierarchical(hierarchy):
    # Phase 1: Optimize leaf nodes
    # Phase 2: Optimize parent arrangements
    # Phase 3: Inter-motif optimization by depth
```

#### 3.3.3 Collision Detection
Using Trimesh for precise mesh-based collision:
```python
def check_collision(mesh1, transform1, mesh2, transform2):
    manager = trimesh.collision.CollisionManager()
    manager.add_object('obj1', mesh1, transform1)
    return manager.in_collision_single(mesh2, transform2)
```

### 3.4 Iterative Generation Strategy

#### 3.4.1 Occupancy-Based Iteration
```python
parameters:
    large_object_generation:
        max_iterations: 2
        target_occupancy_percent: 75.0
    wall_object_generation:
        max_iterations: 1
        target_occupancy_percent: 50.0
    small_object_generation:
        max_iterations: 1
        target_occupancy_percent: 50.0
```

The system iteratively adds objects until:
1. Target occupancy is reached, OR
2. Maximum iterations exceeded, OR
3. No valid placements remain

#### 3.4.2 Surface Saturation Calculation
```python
def calculate_saturation(occupied_area, total_area):
    return (occupied_area / total_area) * 100
    
def should_continue_generation(saturation, target, iteration, max_iter):
    return saturation < target and iteration < max_iter
```

## 4. Key Design Decisions from Code

### 4.1 Motif Selection Logic
- **Object count matching**: Motifs selected based on unique object types
- **Semantic compatibility**: LLM evaluates appropriateness
- **Fallback strategy**: Individual placement if no motif matches

### 4.2 Wall Alignment Mechanism
```python
# Wall detection
def find_nearest_wall(position, room_polygon):
    edges = room_polygon.exterior.coords
    min_distance = float('inf')
    nearest_wall_id = None
    for i, edge in enumerate(edges[:-1]):
        distance = point_to_line_distance(position, edge)
        if distance < min_distance:
            min_distance = distance
            nearest_wall_id = i
    return nearest_wall_id, min_distance

# Alignment application
if motif.wall_alignment:
    position = project_to_wall(position, wall_id)
    rotation = calculate_wall_normal_angle(wall_id)
```

### 4.3 Multi-Stage Validation

**Validation Levels:**
1. **JSON Validation**: Structure and field presence
2. **Semantic Validation**: Object relationships make sense
3. **Geometric Validation**: Dimensions and positions valid
4. **Physical Validation**: Collision and support checks
5. **Contextual Validation**: Fits room type and description

### 4.4 Error Recovery Strategies
- **Retry with feedback**: Up to 5 attempts with specific error messages
- **Constraint relaxation**: Gradually relax collision constraints
- **Fallback generation**: Switch to simpler motifs or individual placement
- **Partial success**: Continue with successfully placed objects

### 4.5 Performance Optimizations
- **Spatial indexing**: R-tree for efficient collision queries
- **Mesh caching**: Reuse loaded 3D models
- **Batch processing**: Process multiple objects simultaneously
- **Early termination**: Stop when constraints cannot be satisfied

## 5. Building a Similar System

### 5.1 Required Components

#### 5.1.1 Core Dependencies
```yaml
# Essential packages
- Python 3.11+
- PyTorch with CUDA
- OpenAI API (GPT-4)
- CLIP model
- Trimesh for 3D operations
- Shapely for 2D geometry
- NumPy, Matplotlib
```

#### 5.1.2 Data Requirements
1. **3D Model Dataset** (~72GB for HSSD)
   - Furniture models in GLB format
   - Decomposed models for part-based objects
   - Support surface annotations

2. **Precomputed Embeddings**
   - CLIP embeddings for all models
   - WordNet synset mappings
   - Category hierarchies

3. **Motif Library**
   - Meta-program definitions
   - Example arrangements
   - Validation patterns

### 5.2 Implementation Steps

#### Step 1: Data Preprocessing
```python
# 1. Extract support surfaces from 3D models
for model in dataset:
    surfaces = extract_support_surfaces(model)
    save_json(surfaces, f"{model_id}_support.json")

# 2. Compute CLIP embeddings
embeddings = []
for model in dataset:
    image = render_model(model)
    embedding = clip_model.encode_image(image)
    embeddings.append(embedding)
save_embeddings(embeddings)

# 3. Build WordNet index
synset_map = {}
for model in dataset:
    synsets = get_wordnet_synsets(model.category)
    synset_map[model.id] = synsets
```

#### Step 2: Prompt Engineering
Create specialized prompts for each stage:
```yaml
system: |
  You are an interior designer specializing in 
  furniture arrangement. Generate realistic layouts
  following spatial constraints and design principles.

populate_large_furniture: |
  Given room: {room_type}
  Description: {description}
  Available furniture: {furniture_list}
  
  Create arrangement with:
  1. Functional groupings (motifs)
  2. Valid positions and rotations
  3. Wall alignments where appropriate
```

#### Step 3: Implement Core Pipeline
```python
class SceneGenerator:
    def generate(self, description):
        # 1. Parse description
        room_type = extract_room_type(description)
        
        # 2. Generate/load room layout
        room = generate_room_layout(description)
        
        # 3. Process each scale
        for scale in ['large', 'wall', 'ceiling', 'small']:
            objects = generate_objects(scale, description)
            motifs = create_motifs(objects)
            optimized = optimize_placement(motifs, room)
            scene.add(optimized)
            
        return scene
```

#### Step 4: Optimization Tuning

**Key Parameters to Tune:**
```python
# Spatial solver
grid_size: 0.05-0.2  # Finer = more precise but slower
collision_threshold: 0.0-0.05  # Small overlap tolerance

# Generation control  
temperature: 0.5-0.9  # LLM creativity
top_k_retrieval: 5-20  # Retrieval candidates
occupancy_target: 50-90  # Density control

# Optimization weights
wall_alignment_weight: 1.0-5.0
initial_placement_weight: 1.0-10.0
collision_penalty: 10.0-100.0
```

### 5.3 Critical Implementation Insights

#### 5.3.1 Motif Decomposition Strategy
The key to realistic scenes is proper motif decomposition:
1. Always identify primary functional group first
2. Handle remaining objects as secondary arrangements
3. Use compositional motifs to combine sub-arrangements
4. Fallback to individual placement only when necessary

#### 5.3.2 Support Surface Management
```python
# Maintain support surface availability
class SupportManager:
    def __init__(self):
        self.surfaces = {}  # surface_id -> available_area
        
    def place_object(self, obj, surface_id):
        # Check available space
        if self.get_available_area(surface_id) < obj.footprint:
            return False
        # Update occupied area
        self.surfaces[surface_id] -= obj.footprint
        return True
```

#### 5.3.3 Hierarchical Dependencies
```python
# Ensure proper dependency order
dependency_graph = {
    'small': ['large', 'wall'],  # Small depends on large and wall
    'ceiling': [],  # Independent
    'wall': [],  # Independent  
    'large': []  # Independent
}

# Process in topological order
processing_order = topological_sort(dependency_graph)
```

#### 5.3.4 Validation Functions
Essential validation checks:
```python
def validate_arrangement(arrangement, room):
    # 1. Boundary check
    if not room.contains(arrangement.bbox):
        return False, "Outside room boundary"
    
    # 2. Height check
    if arrangement.max_height > room.ceiling_height:
        return False, "Exceeds ceiling height"
    
    # 3. Support check
    if not has_valid_support(arrangement):
        return False, "Lacks proper support"
        
    # 4. Accessibility check
    if blocks_doorway(arrangement, room.door):
        return False, "Blocks doorway"
        
    return True, "Valid"
```

### 5.4 Advanced Features

#### 5.4.1 Style Transfer
```python
# Extract style from reference scenes
def extract_style(reference_scene):
    return {
        'color_palette': extract_colors(reference_scene),
        'object_styles': extract_object_categories(reference_scene),
        'density': calculate_density(reference_scene),
        'arrangement_patterns': extract_motifs(reference_scene)
    }
```

#### 5.4.2 Interactive Refinement
```python
# Allow user to refine generated scenes
def refine_scene(scene, user_feedback):
    if "move" in user_feedback:
        object_id, new_position = parse_move(user_feedback)
        scene.move_object(object_id, new_position)
        
    if "replace" in user_feedback:
        object_id, new_type = parse_replace(user_feedback)
        new_object = retrieve_object(new_type)
        scene.replace_object(object_id, new_object)
```

#### 5.4.3 Quality Metrics
```python
# Evaluate scene quality
def evaluate_scene(scene):
    metrics = {
        'realism': compute_realism_score(scene),
        'functionality': check_functional_constraints(scene),
        'aesthetics': evaluate_visual_balance(scene),
        'diversity': measure_object_variety(scene),
        'clutter': calculate_clutter_score(scene)
    }
    return metrics
```

## 6. Detailed Motif System Implementation

### 6.1 Motif Definition Structure

Based on the code analysis, motifs are defined with specific constraints and spatial programs:

```yaml
# Example: Dining Set Motif
dining_set:
  description: "A dining table surrounded by chairs with proper spacing"
  constraints:
    min_objects: 2
    max_objects: 8
    required_types: ["table", "chair"]
    spatial_requirements:
      - chairs_must_face_table: true
      - maintain_spacing: 0.6-1.0  # meters
      - allow_asymmetric: true
  
  program_template: |
    def dining_set(table, chairs):
        arrangement = create_arrangement()
        center_position = (0, 0, 0)
        
        # Place table at center
        table_obj = create(table.type, table.dimensions)
        move(table_obj, *center_position)
        arrangement.add(table_obj)
        
        # Calculate chair positions
        num_chairs = len(chairs)
        spacing = calculate_optimal_spacing(table, chairs[0])
        
        for i, chair in enumerate(chairs):
            angle = (360 / num_chairs) * i
            position = calculate_perimeter_position(center_position, spacing, angle)
            chair_obj = create(chair.type, chair.dimensions)
            move(chair_obj, *position)
            rotate(chair_obj, 'y', angle + 180)  # Face the table
            arrangement.add(chair_obj)
            
        return arrangement
```

### 6.2 Motif Selection Algorithm

```python
def select_motif(object_specs, description):
    # 1. Count unique object types
    unique_types = set(spec.category for spec in object_specs)
    type_counts = {t: sum(1 for spec in object_specs if spec.category == t) 
                   for t in unique_types}
    
    # 2. Filter motifs by constraints
    candidate_motifs = []
    for motif in MOTIF_LIBRARY:
        if motif.matches_constraints(type_counts):
            semantic_score = calculate_semantic_match(motif, description)
            candidate_motifs.append((motif, semantic_score))
    
    # 3. Select best matching motif
    if candidate_motifs:
        return max(candidate_motifs, key=lambda x: x[1])[0]
    
    # 4. Fallback to individual placement
    return create_individual_motifs(object_specs)
```

### 6.3 Program Execution Engine

The system uses a custom interpreter for spatial programs:

```python
class SpatialProgramExecutor:
    def __init__(self):
        self.globals = {
            'create': self._create_object,
            'move': self._move_object,
            'rotate': self._rotate_object,
            'face_to_face': self._face_to_face,
            'surround': self._surround,
            'stack': self._stack,
            'align_to_wall': self._align_to_wall
        }
    
    def execute(self, program_code, variables):
        """Execute spatial program with given variables"""
        local_vars = variables.copy()
        exec(program_code, self.globals, local_vars)
        return local_vars.get('arrangement', None)
    
    def _create_object(self, obj_type, dimensions):
        """Create object with bounding box"""
        return Obj(label=obj_type, 
                  bounding_box=BoundingBox(half_size=np.array(dimensions)/2))
    
    def _move_object(self, obj, x, y, z):
        """Move object to position"""
        obj.bounding_box.centroid = np.array([x, y, z])
    
    def _rotate_object(self, obj, axis, degrees):
        """Rotate object around axis"""
        radians = np.deg2rad(degrees)
        if axis == 'y':
            rotation_matrix = rotation_matrix_y(radians)
            obj.bounding_box.coord_axes = obj.bounding_box.coord_axes @ rotation_matrix
```

## 7. Support Surface Analysis Deep Dive

### 7.1 Surface Extraction Pipeline

The support surface analysis is crucial for placing small objects:

```python
def extract_support_surfaces(mesh_path):
    """Extract support surfaces from 3D mesh"""
    mesh = trimesh.load(mesh_path)
    
    # 1. Identify horizontal faces
    horizontal_faces = []
    for face_idx, normal in enumerate(mesh.face_normals):
        if normal[1] > NORMAL_HORIZONTAL_THRESHOLD:  # Y-up convention
            horizontal_faces.append(face_idx)
    
    # 2. Group faces by height
    face_heights = {}
    for face_idx in horizontal_faces:
        vertices = mesh.vertices[mesh.faces[face_idx]]
        height = np.mean(vertices[:, 1])
        height_key = round(height, 4)
        
        if height_key not in face_heights:
            face_heights[height_key] = []
        face_heights[height_key].append(face_idx)
    
    # 3. Create support regions for each height
    support_regions = {}
    for height, face_indices in face_heights.items():
        # Merge overlapping faces
        merged_geometry = merge_faces_to_polygon(mesh, face_indices)
        
        # Filter by area
        if merged_geometry.area > MIN_AREA:
            support_regions[height] = {
                'geometry': merged_geometry,
                'area': merged_geometry.area,
                'height': height,
                'available_space_above': calculate_clearance(mesh, height)
            }
    
    return support_regions

def calculate_clearance(mesh, surface_height):
    """Calculate available space above surface"""
    # Cast rays upward from surface to find obstacles
    clearance = TOP_HEIGHT  # Default ceiling height
    
    for vertex in mesh.vertices:
        if abs(vertex[1] - surface_height) < 0.01:  # On surface
            # Cast ray upward
            ray_origin = vertex + [0, 0.01, 0]  # Slightly above surface
            ray_direction = [0, 1, 0]  # Upward
            
            intersections = mesh.ray.intersects_location([ray_origin], [ray_direction])
            if len(intersections[0]) > 0:
                hit_height = intersections[0][0][1]
                clearance = min(clearance, hit_height - surface_height)
    
    return clearance
```

### 7.2 Surface Quality Assessment

```python
def assess_surface_quality(support_region):
    """Assess suitability of surface for object placement"""
    scores = {}
    
    # Area score
    scores['area'] = min(1.0, support_region['area'] / 1.0)  # Normalize to 1 m²
    
    # Shape score (prefer regular shapes)
    geometry = support_region['geometry']
    bbox_area = geometry.bounds[2] * geometry.bounds[3]
    shape_regularity = geometry.area / bbox_area
    scores['shape'] = shape_regularity
    
    # Accessibility score (not blocked by obstacles)
    accessibility = calculate_accessibility(support_region)
    scores['accessibility'] = accessibility
    
    # Stability score (flat and stable)
    scores['stability'] = 1.0 if support_region['available_space_above'] > 0.3 else 0.5
    
    # Combined score
    weights = {'area': 0.3, 'shape': 0.2, 'accessibility': 0.3, 'stability': 0.2}
    total_score = sum(scores[key] * weights[key] for key in scores)
    
    return total_score, scores
```

## 8. Spatial Optimization Details

### 8.1 DFS Solver Implementation

```python
class DFSSolver:
    def _dfs_search(self, remaining_motifs, current_solution, depth, verbose):
        """Recursive DFS search for motif placement"""
        
        # Base case: all motifs placed
        if not remaining_motifs:
            score = self._evaluate_solution(current_solution)
            self.solutions.append({
                'placements': current_solution.copy(),
                'score': score
            })
            if verbose:
                print(f"Solution found with score: {score:.3f}")
            return
        
        # Timeout check
        if time.time() - self.start_time > self.config.max_duration:
            if verbose:
                print("Search timeout reached")
            return
        
        # Get next motif to place
        current_motif = remaining_motifs[0]
        motif_name, dimensions, constraints = current_motif
        
        # Generate candidate positions
        candidates = self._generate_candidates(current_motif, current_solution)
        
        # Try each candidate
        for candidate in candidates[:self.config.max_candidates_per_motif]:
            if self._is_valid_placement(candidate, current_solution):
                # Recursive call
                new_solution = current_solution.copy()
                new_solution[motif_name] = candidate
                
                self._dfs_search(remaining_motifs[1:], new_solution, depth + 1, verbose)
    
    def _generate_candidates(self, motif, current_solution):
        """Generate candidate positions for motif"""
        motif_name, dimensions, constraints = motif
        candidates = []
        
        # Get grid points within surface
        grid_points = self._get_grid_points(dimensions)
        
        # Generate rotations
        rotations = [0, 90, 180, 270] if self._allows_rotation(constraints) else [0]
        
        for point in grid_points:
            for rotation in rotations:
                candidate = MotifPlacement(
                    center_x=point[0],
                    center_y=point[1], 
                    rotation=rotation,
                    bbox=self._calculate_bbox(point, dimensions, rotation),
                    score=self._score_position(point, constraints, current_solution)
                )
                candidates.append(candidate)
        
        # Sort by score (best first)
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates
    
    def _score_position(self, position, constraints, current_solution):
        """Score a candidate position"""
        score = 0.0
        
        # Initial placement preference
        if hasattr(self, 'initial_placements'):
            initial_pos = self.initial_placements.get(constraints.get('motif_id'))
            if initial_pos:
                distance = np.linalg.norm(np.array(position) - np.array([initial_pos.center_x, initial_pos.center_y]))
                score += self.config.initial_placement_weight * np.exp(-distance / self.config.initial_placement_range)
        
        # Wall alignment preference
        if constraints.get('wall_alignment'):
            wall_score = self._calculate_wall_alignment_score(position)
            score += self.config.wall_alignment_weight * wall_score
        
        # Avoid crowded areas
        crowding_penalty = self._calculate_crowding_penalty(position, current_solution)
        score -= crowding_penalty
        
        return score
```

### 8.2 Collision Detection System

```python
class CollisionDetector:
    def __init__(self):
        self.collision_manager = trimesh.collision.CollisionManager()
        self.object_meshes = {}
    
    def add_object(self, obj_id, mesh, transform):
        """Add object to collision detection"""
        self.collision_manager.add_object(obj_id, mesh, transform)
        self.object_meshes[obj_id] = (mesh, transform)
    
    def check_collision(self, new_mesh, new_transform, exclude_ids=None):
        """Check if new object collides with existing objects"""
        exclude_ids = exclude_ids or []
        
        # Temporarily remove excluded objects
        temp_removed = {}
        for obj_id in exclude_ids:
            if obj_id in self.object_meshes:
                temp_removed[obj_id] = self.object_meshes[obj_id]
                self.collision_manager.remove_object(obj_id)
        
        # Check collision
        is_colliding = self.collision_manager.in_collision_single(new_mesh, new_transform)
        
        # Restore excluded objects
        for obj_id, (mesh, transform) in temp_removed.items():
            self.collision_manager.add_object(obj_id, mesh, transform)
        
        return is_colliding
    
    def get_collision_pairs(self):
        """Get all colliding object pairs"""
        colliding_pairs = []
        object_ids = list(self.object_meshes.keys())
        
        for i in range(len(object_ids)):
            for j in range(i + 1, len(object_ids)):
                obj1_id, obj2_id = object_ids[i], object_ids[j]
                mesh1, transform1 = self.object_meshes[obj1_id]
                mesh2, transform2 = self.object_meshes[obj2_id]
                
                if self._meshes_collide(mesh1, transform1, mesh2, transform2):
                    colliding_pairs.append((obj1_id, obj2_id))
        
        return colliding_pairs
```

## 9. Validation System Architecture

### 9.1 Multi-Level Validation

```python
class ValidationPipeline:
    def __init__(self):
        self.validators = [
            JSONStructureValidator(),
            SemanticValidator(),
            GeometricValidator(),
            PhysicalValidator(),
            ContextualValidator()
        ]
    
    def validate(self, response, context):
        """Run full validation pipeline"""
        for validator in self.validators:
            is_valid, error_msg, error_idx = validator.validate(response, context)
            if not is_valid:
                return is_valid, error_msg, error_idx
        return True, "Valid", 0

class JSONStructureValidator:
    def validate(self, response, context):
        """Validate JSON structure and required fields"""
        try:
            data = json.loads(extract_json(response))
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}", 0
        
        required_fields = context.get('required_fields', [])
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}", 0
        
        return True, "Valid JSON structure", 0

class SemanticValidator:
    def validate(self, response, context):
        """Validate semantic consistency"""
        data = json.loads(extract_json(response))
        
        # Check object ID consistency
        if 'motifs' in data:
            referenced_objects = set()
            available_objects = set(context.get('object_ids', []))
            
            for motif in data['motifs']:
                for obj_ref in motif.get('objects', []):
                    referenced_objects.add(obj_ref)
            
            missing_objects = referenced_objects - available_objects
            if missing_objects:
                return False, f"Referenced undefined objects: {missing_objects}", 0
        
        return True, "Semantically valid", 0

class GeometricValidator:
    def validate(self, response, context):
        """Validate geometric constraints"""
        data = json.loads(extract_json(response))
        room_polygon = context.get('room_polygon')
        
        if 'positions' in data and room_polygon:
            for item in data['positions']:
                pos = item.get('position', [0, 0])
                point = Point(pos[0], pos[2])  # x, z coordinates
                
                if not room_polygon.contains(point):
                    return False, f"Object {item.get('id')} outside room boundary", 0
        
        return True, "Geometrically valid", 0

class PhysicalValidator:
    def validate(self, response, context):
        """Validate physical plausibility"""
        data = json.loads(extract_json(response))
        
        # Check support relationships
        if 'arrangements' in data:
            for arrangement in data['arrangements']:
                if not self._has_valid_support(arrangement, context):
                    return False, f"Arrangement lacks proper support", 0
        
        return True, "Physically plausible", 0
```

## 10. Prompt Engineering Strategies

### 10.1 Stage-Specific Prompts

The system uses carefully crafted prompts for each generation stage:

```yaml
# scene_prompts_large.yaml
system: |
  You are an expert interior designer specializing in furniture arrangement.
  Your task is to create realistic, functional layouts that follow design principles
  and spatial constraints. Always consider traffic flow, functionality, and aesthetics.

populate_surface_motifs: |
  You are arranging large furniture in a {room_type}.
  
  ROOM DETAILS:
  {room_details}
  
  AVAILABLE FURNITURE:
  {large_furniture}
  
  TASK: Create functional groupings (motifs) that:
  1. Use appropriate spatial relationships
  2. Consider traffic flow and accessibility
  3. Follow interior design principles
  4. Create cohesive functional areas
  
  Return JSON with motifs list containing:
  - id: unique identifier
  - motif_type: from predefined types
  - description: functional purpose
  - objects: furniture assignments
  - constraints: spatial requirements

populate_room_provided: |
  Given these MOTIFS and room layout image:
  {MOTIFS}
  
  Determine optimal positions and rotations for each motif:
  1. Avoid blocking doorways
  2. Maintain proper clearances
  3. Consider wall alignments
  4. Ensure accessibility
  
  Return JSON with positions list containing:
  - id: motif identifier  
  - position: [x, z] coordinates
  - rotation: degrees (0-359)
  - wall_alignment: boolean
  - ignore_collision: boolean (only if necessary)
```

### 10.2 Validation Feedback Prompts

```yaml
# Error-specific feedback prompts
populate_surface_motifs_feedback_0: |
  ERROR: Your motif selection doesn't match the available furniture.
  
  RULES:
  - Each furniture piece must be assigned to exactly one motif
  - Motif types must match object constraints
  - Consider functional relationships between objects
  
  Please revise your motif assignments.

populate_room_provided_feedback_0: |
  ERROR: Object positions are outside room boundaries or blocking access.
  
  CONSTRAINTS:
  - All objects must be within room polygon
  - Maintain clear paths to doors
  - Ensure adequate spacing between furniture
  
  Please adjust positions accordingly.
```

### 10.3 Context-Aware Prompting

```python
def build_context_prompt(stage, scene_state, user_description):
    """Build context-aware prompt for generation stage"""
    
    context = {
        'room_type': extract_room_type(user_description),
        'style_preferences': extract_style_cues(user_description),
        'existing_objects': get_existing_objects(scene_state),
        'available_space': calculate_available_space(scene_state),
        'constraints': get_active_constraints(scene_state)
    }
    
    # Add stage-specific context
    if stage == 'large':
        context.update({
            'floor_area': scene_state.room_polygon.area,
            'room_shape': classify_room_shape(scene_state.room_polygon),
            'door_locations': scene_state.door_location,
            'window_locations': scene_state.window_locations
        })
    
    elif stage == 'small':
        context.update({
            'support_surfaces': get_available_surfaces(scene_state),
            'surface_types': categorize_surfaces(scene_state),
            'existing_motifs': [m.to_gpt_dict() for m in scene_state.get_all_motifs()]
        })
    
    return context
```

## 11. Error Recovery and Robustness

### 11.1 Graceful Degradation

```python
class RobustSceneGenerator:
    def __init__(self):
        self.fallback_strategies = [
            'retry_with_relaxed_constraints',
            'use_simpler_motifs',
            'individual_placement',
            'partial_generation'
        ]
    
    def generate_with_fallbacks(self, description, max_attempts=3):
        """Generate scene with fallback strategies"""
        
        for attempt in range(max_attempts):
            try:
                # Try normal generation
                scene = self.generate_scene(description)
                return scene
                
            except GenerationFailure as e:
                print(f"Generation failed (attempt {attempt + 1}): {e}")
                
                # Apply fallback strategy
                if attempt < len(self.fallback_strategies):
                    strategy = self.fallback_strategies[attempt]
                    description = self._apply_fallback(strategy, description, e)
                else:
                    # Final fallback: minimal scene
                    return self._create_minimal_scene(description)
    
    def _apply_fallback(self, strategy, description, error):
        """Apply specific fallback strategy"""
        
        if strategy == 'retry_with_relaxed_constraints':
            # Reduce collision sensitivity
            self.config.collision_threshold += 0.02
            # Reduce occupancy targets
            self.config.target_occupancy *= 0.8
            
        elif strategy == 'use_simpler_motifs':
            # Prefer individual motifs over complex arrangements
            self.motif_selector.prefer_simple = True
            
        elif strategy == 'individual_placement':
            # Disable motif system, place objects individually
            self.config.use_scene_motifs = False
            
        elif strategy == 'partial_generation':
            # Generate only essential objects
            self.config.object_types = ['large']  # Only furniture
        
        return description
```

### 11.2 Constraint Relaxation

```python
class ConstraintRelaxer:
    def __init__(self):
        self.relaxation_stages = [
            {'collision_threshold': 0.01, 'occupancy_reduction': 0.9},
            {'collision_threshold': 0.03, 'occupancy_reduction': 0.8},
            {'collision_threshold': 0.05, 'occupancy_reduction': 0.7},
            {'disable_wall_alignment': True, 'occupancy_reduction': 0.6}
        ]
    
    def relax_constraints(self, stage_idx, solver_config):
        """Apply progressive constraint relaxation"""
        
        if stage_idx >= len(self.relaxation_stages):
            return False  # No more relaxation possible
        
        relaxation = self.relaxation_stages[stage_idx]
        
        # Update solver configuration
        if 'collision_threshold' in relaxation:
            solver_config.collision_threshold = relaxation['collision_threshold']
        
        if 'occupancy_reduction' in relaxation:
            solver_config.target_occupancy *= relaxation['occupancy_reduction']
        
        if 'disable_wall_alignment' in relaxation:
            solver_config.enforce_wall_alignment = False
        
        return True
```

## 12. Performance Optimization

### 12.1 Caching Strategies

```python
class SceneGenerationCache:
    def __init__(self):
        self.mesh_cache = {}
        self.embedding_cache = {}
        self.surface_cache = {}
        self.arrangement_cache = {}
    
    def get_mesh(self, mesh_path):
        """Cached mesh loading"""
        if mesh_path not in self.mesh_cache:
            mesh = trimesh.load(mesh_path)
            self.mesh_cache[mesh_path] = mesh
        return self.mesh_cache[mesh_path]
    
    def get_embeddings(self, model_id):
        """Cached CLIP embeddings"""
        if model_id not in self.embedding_cache:
            embedding = self._compute_embedding(model_id)
            self.embedding_cache[model_id] = embedding
        return self.embedding_cache[model_id]
    
    def get_support_surfaces(self, mesh_path):
        """Cached support surface analysis"""
        if mesh_path not in self.surface_cache:
            surfaces = extract_support_surfaces(mesh_path)
            self.surface_cache[mesh_path] = surfaces
        return self.surface_cache[mesh_path]
```

### 12.2 Parallel Processing

```python
async def process_motifs_parallel(motifs, model_manager):
    """Process multiple motifs in parallel"""
    
    # Create tasks for parallel processing
    tasks = []
    for motif in motifs:
        task = asyncio.create_task(
            decompose_motif_async(motif, model_manager)
        )
        tasks.append(task)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results and handle exceptions
    successful_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Motif {motifs[i].id} failed: {result}")
        else:
            successful_results.append(result)
    
    return successful_results
```

## 13. Conclusion

HSM represents a significant advance in indoor scene generation through its hierarchical, motif-based approach. The key technical innovations include:

1. **Hierarchical Processing**: Multi-scale generation ensuring proper dependencies between object scales
2. **Scene Motifs**: Capturing recurring spatial patterns for realistic object arrangements  
3. **LLM Integration**: Using language models for semantic understanding and spatial reasoning
4. **Robust Optimization**: Multiple solvers and fallback strategies for reliable placement
5. **Validation Pipeline**: Multi-level validation ensuring physical and semantic plausibility

The system's modular architecture allows for improvements and extensions at each level, making it a solid foundation for indoor scene generation research and applications.

## Appendix: Configuration Examples

### A.1 Complete Scene Configuration
```yaml
room:
  room_description: "A modern living room with L-shaped sofa and coffee table"
  vertices: null  # Auto-generate room layout
  door_location: null
  height: 3.0
  window_locations: [[2,0,1,1]]  # [wall_index, position, width, height]

mode:
  use_scene_motifs: true
  use_solver: true
  enable_spatial_optimization: true
  object_types: ["large", "wall", "ceiling", "small"]
  extra_types: ["large", "wall", "ceiling", "small"]

execution:
  result_dir: "results/living_room"

parameters:
  large_object_generation:
    max_iterations: 2
    target_occupancy_percent: 75.0
  wall_object_generation:
    max_iterations: 1
    target_occupancy_percent: 50.0
  ceiling_object_generation:
    max_iterations: 1
    target_occupancy_percent: 50.0
  small_object_generation:
    max_iterations: 1
    target_occupancy_percent: 50.0
```

### A.2 Solver Configuration
```python
@dataclass
class DFSSolverConfig:
    grid_size: float = 0.1
    max_duration: float = 10.0
    max_candidates_per_motif: int = 10
    alignment_threshold: float = -0.7
    epsilon: float = 1e-6
    
    # Soft constraint weights
    initial_placement_range: float = 5.0
    initial_placement_weight: float = 5.0
    wall_alignment_weight: float = 2.5
    wall_alignment_range: float = 0.5
```

### A.3 Example Motif Implementation
```python
def create_seating_area(sofa, coffee_table, side_tables=None):
    """Create a seating area motif"""
    arrangement = Arrangement([], "Cozy seating area with sofa and coffee table")
    
    # Place sofa against wall
    sofa_obj = create(sofa.type, sofa.dimensions)
    move(sofa_obj, 0, 0, -1.5)  # Against back wall
    rotate(sofa_obj, 'y', 0)  # Facing forward
    arrangement.objs.append(sofa_obj)
    
    # Place coffee table in front of sofa
    table_obj = create(coffee_table.type, coffee_table.dimensions)
    sofa_front = sofa.dimensions[2] / 2 + coffee_table.dimensions[2] / 2 + 0.6
    move(table_obj, 0, 0, -1.5 + sofa_front)
    arrangement.objs.append(table_obj)
    
    # Add side tables if available
    if side_tables:
        for i, side_table in enumerate(side_tables[:2]):  # Max 2 side tables
            side_obj = create(side_table.type, side_table.dimensions)
            x_offset = (i * 2 - 1) * (sofa.dimensions[0] / 2 + side_table.dimensions[0] / 2 + 0.2)
            move(side_obj, x_offset, 0, -1.5)
            arrangement.objs.append(side_obj)
    
    return arrangement
```

This comprehensive documentation provides the complete technical foundation needed to understand, implement, and extend the HSM system for hierarchical indoor scene generation.