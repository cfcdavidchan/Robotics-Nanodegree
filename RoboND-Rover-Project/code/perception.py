import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# Define a function to convert to rover-centric coordinates
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = np.absolute(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[0]).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to apply a rotation to pixel positions
def rotate_pix(xpix, ypix, yaw):
    # TODO:
    # Convert yaw to radians
    # Apply a rotation
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = xpix * np.cos(yaw_rad) - ypix * np.sin(yaw_rad)
    ypix_rotated = xpix * np.sin(yaw_rad) + ypix * np.cos(yaw_rad)
    # Return the result  
    return xpix_rotated, ypix_rotated

# Define a function to perform a translation
def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # TODO:
    # Apply a scaling and a translation
    scale = 10
    # Perform translation and convert to integer since pixel values can't be float
    xpix_translated = np.int_(xpos + (xpix_rot / scale))
    ypix_translated = np.int_(ypos + (ypix_rot / scale))
    # Return the result  
    return xpix_translated, ypix_translated

# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped

def rock_thresh(img, boundary =([100,100,0], [200,200,70])):
    """apply thresholding to find the rock sample"""
    lower = np.array(boundary[0], dtype = "uint8")
    upper = np.array(boundary[1], dtype = "uint8")
    # create mask
    mask = cv2.inRange(img, lower, upper)
    # apply image masking
    output = cv2.bitwise_and(img, img, mask = mask)
    # convert result to gray
    output = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
    # create temp image for our rock
    zeros = np.zeros_like(img[:,:,0])
    try:
        # get the closest coordinate of the transformed rock
        closest_x = max(output.nonzero()[1])
        closest_y = max(output.nonzero()[0])
        # make the rock look bigger instead of just dot
        zeros[closest_y:closest_y+5,closest_x:closest_x+5] = 1
        return zeros
    except:
        #if no rock is in image, return zeros
        return zeros
# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    dst_size = 5 
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - bottom_offset],
                  [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - bottom_offset],
                  [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset], 
                  [Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset],
                  ])
    # 2) Apply perspective transform
    warped_terrain = perspect_transform(Rover.img, source, destination)

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    thresholded_navigable = color_thresh(warped_terrain, rgb_thresh=(160, 160, 160))

    # get index of the navigable terrain
    not_obstacle_index = thresholded_navigable.nonzero()
    # create obstacle image, which is the reverse of navigable terrain (thresholded)
    obstacle = np.ones_like(Rover.img[:,:,0])
    obstacle[not_obstacle_index] = 0

    # detect rock if it exist
    rock = rock_thresh(warped_terrain)

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)

    # update Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
    Rover.vision_image[:,:,0] = obstacle

    # update Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
    Rover.vision_image[:,:,1] = rock

    # update Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
    Rover.vision_image[:,:,2] = thresholded_navigable

    # 5) Convert map image pixel values to rover-centric coords
    navigable_xpix, navigable_ypix = rover_coords(thresholded_navigable)
    obstacle_xpix, obstacle_ypix = rover_coords(obstacle)
    rock_xpix, rock_ypix = rover_coords(rock)

    # 6) Convert rover-centric pixel values to world coordinates
    world_size = 200
    scale = 10
    navigable_xpix_world, navigable_ypix_world = pix_to_world(navigable_xpix, navigable_ypix, Rover.pos[0], Rover.pos[1], 
                                                      Rover.yaw, world_size, scale)
    obstacle_xpix_world, obstacle_ypix_world = pix_to_world(obstacle_xpix, obstacle_ypix, Rover.pos[0], Rover.pos[1], 
                                                      Rover.yaw, world_size, scale)
    rock_xpix_world, rock_ypix_world = pix_to_world(rock_xpix, rock_ypix, Rover.pos[0], Rover.pos[1], 
                                                      Rover.yaw, world_size, scale)

    # 7) Update Rover worldmap (to be displayed on right side of screen)
    Rover.worldmap[obstacle_ypix_world, obstacle_xpix_world, 0] += 1
    Rover.worldmap[rock_ypix_world, rock_xpix_world, 1] += 1
    Rover.worldmap[navigable_ypix_world, navigable_xpix_world, 2] += 1


    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
    dist, angles = to_polar_coords(navigable_xpix, navigable_ypix)
    Rover.nav_dists = dist
    Rover.nav_angles = angles
      
    
    return Rover