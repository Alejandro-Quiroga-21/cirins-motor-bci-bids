import pygame
import random
import os
import time
import cairosvg
from screeninfo import get_monitors
import winsound

#miembro = 'Inferior'
miembro = 'Superior'

lateralidad = 'right'
#lateralidad = 'left'

# Cargar la secuencia de imágenes
# Animation frames as deposited in Dataset_C/stimuli/<foot|hand>_<left|right>/
image_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Dataset_C", "stimuli",
                            {"Inferior": "foot", "Superior": "hand"}[miembro] + "_" + lateralidad)
images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])

def get_secondary_monitor():
    monitors = get_monitors()
    if len(monitors) > 1:
        return monitors[1]
    return monitors[0]  # Si no hay segunda pantalla, usar la principal

def generacion_aleatoria(estimulos_clase, tiempo_estimulacion):
    LEFT = (255, 255, 255)
    PAUSA = (63,72,204)

    balanced_list = [0] * estimulos_clase 

    # Crear una lista intercalada con el número 1 entre cada elemento
    intercalated_list = []
    time_stimulation = []
    sound_time = []
    for i in range(len(balanced_list)):
        intercalated_list.append(balanced_list[i])
        time_stimulation.append(tiempo_estimulacion)
        sound_time.append(100)
        if i != len(balanced_list) - 1:
            intercalated_list.append(1)
            time_stimulation.append(round(random.uniform(5, 6),1))
            sound_time.append(0)

    intercalated_list.insert(0, 1)
    time_stimulation.insert(0, 3)
    sound_time.insert(0,0)
    intercalated_list.append(1)
    time_stimulation.append(3)
    sound_time.append(0)
    colors_trigger = []
    for i in intercalated_list:
        if i == 1:
            colors_trigger.append(PAUSA)
        elif i == 0:
            colors_trigger.append(LEFT)
    print(time_stimulation)
    return intercalated_list, time_stimulation, colors_trigger, sound_time

def run_animation_once(screen, clock, first_frame):
    current_frame = 0
    total_frames = len(images)
    WIDTH, HEIGHT = screen.get_size()

    while current_frame < total_frames:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        # Cargar la imagen actual
        frame = pygame.image.load(os.path.join(image_folder, images[current_frame]))
        frame_width, frame_height = frame.get_size()

        # Calcular la nueva escala para llenar la pantalla
        scale = max(WIDTH / frame_width, HEIGHT / frame_height)
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)
        frame = pygame.transform.scale(frame, (new_width, new_height))

        # Limpiar la pantalla, mostrar el primer fotograma y dibujar el cuadro blanco
        screen.blit(first_frame, (0, 0))
        #pygame.draw.rect(screen, (255, 255, 255), (0, HEIGHT - 80, 70, 80))  # Dibuja el cuadro blanco
        screen.blit(frame, (0, 0))

        # Actualizar el índice del frame
        current_frame += 1
        pygame.draw.rect(screen, (255, 255, 255), (0, HEIGHT - 80, 70, 80))
        # Actualizar la pantalla
        pygame.display.flip()

        # Controlar la velocidad de la animación
        clock.tick(24)  # 24 frames por segundo

def mostrar_estimulo(screen, tiempo_visualizacion, index, color, sound, clock, first_frame):
    screen.blit(first_frame, (0, 0))  # Mostrar el primer fotograma
    pygame.draw.rect(screen, color, (0, screen.get_height() - 80, 70, 80))  # Dibuja el cuadro
    pygame.display.flip()
    winsound.Beep(1000, sound)  # 1000 Hz, 'duration' ms
    time.sleep(tiempo_visualizacion)
    print(tiempo_visualizacion)

    if sound == 100:
        run_animation_once(screen, clock, first_frame)  # Ejecutar la animación completa

def main():
    pygame.init()

    secondary_monitor = get_secondary_monitor()
    WIDTH, HEIGHT = secondary_monitor.width, secondary_monitor.height
    WINDOW_SIZE = (WIDTH, HEIGHT)
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{secondary_monitor.x},{secondary_monitor.y}"

    screen = pygame.display.set_mode(WINDOW_SIZE)
    clock = pygame.time.Clock()

    # Cargar y redimensionar el primer fotograma
    first_frame = pygame.image.load(os.path.join(image_folder, images[0]))
    frame_width, frame_height = first_frame.get_size()
    scale = max(WIDTH / frame_width, HEIGHT / frame_height)
    new_width = int(frame_width * scale)
    new_height = int(frame_height * scale)
    first_frame = pygame.transform.scale(first_frame, (new_width, new_height))

    Estimulos_clase = 10
    stim_list, time_stim, color_triggers, sound_time = generacion_aleatoria(Estimulos_clase, 2.5)  

    index = 0
    running = True
    while running:
        screen.blit(first_frame, (0, 0))  # Mostrar el primer fotograma por defecto
        pygame.draw.rect(screen, (255, 255, 255), (0, HEIGHT - 80, 70, 80))  # Dibuja el cuadro blanco
        pygame.display.flip()

        if index == len(stim_list):
            break
        mostrar_estimulo(screen, time_stim[index], stim_list[index], color_triggers[index], sound_time[index], clock, first_frame)
        index += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()

if __name__ == "__main__":
    main()
