
class Gesture:
    """
    This is a base class for a single gesture
    The start_ms and end_ms are the milliseconds that the gesture takes place in
    The mouth_part (constriction ?? as originally named) is supposed to describe which
    part of the mouth the gesture takes place in
    The degree is the degree to which the distances in the mouth parts are closed
    or opened
    The stiffness describes the sharpness of the closure
    """

    # Bilabial = 0
    # Alveolar = 1
    # Palatal = 2
    # Velar = 3
    # Pharyngeal = 4
    # Velopharyngeal = 5

    # These are the corresponding codes

    def __init__(self, start_ms, end_ms, mouth_part, degree, stiffness):
        self.start_s = float(start_ms.strip()) / 1000
        self.start_ms = int(float(start_ms))
        self.end_s = float(end_ms.strip()) / 1000
        self.end_ms = int(float(end_ms))
        self.mouth_part = int(mouth_part.strip())
        self.degree = round(float(degree.strip()), 2)
        # print(self.degree)
        self.stiffness = round(float(stiffness.strip()), 2)


    def __str__(self):
        return f'{self.start_ms} {self.end_ms} {self.mouth_part} {self.degree} {self.stiffness}'
