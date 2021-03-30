import heapq

# Se crean las clases esenciales


class State:
    """
    Clase que representa el estado del canal
    """
    def __init__(self, x):
        """
        :param x: tamaño del buffer de estados
        """
        self.s = []
        self.tam = x
        for i in range(x):
            self.s.append(0)

    def push(self, val):
        """
        :param val: estado del canal en ese instante de tiempo
                  puede valer 0 (Canal libre)
                  puede valer 1 (Canal ocupado)
                  puede valer 2 (Colisión)
        """
        for i in range(1, self.tam):
            self.s[i - 1] = self.s[i]
        self.s[-1] = val

    def get_state(self):
        return self.s


class Event:
    def __init__(self, tiempo, tipo, nodo, emisor=None, l_pkt=None):
        self.tiempo = tiempo
        self.tipo = tipo
        self.nodo = nodo
        self.emisor = emisor
        self.len_pkt = l_pkt


class Sim:
    """
    Clase del simulador
    """
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._queue, (priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]

    def size(self):
        return len(self._queue)


class Canal:
    """
    Clase que representa al canal
    """
    def __init__(self):
        self.n_trans = 0
        self.col = False

    def estado(self, tipo=False):
        """
        :param tipo: Nos dice si queremos que el canal pueda diferenciar entre colisiones y canal ocupado (True) o no (False)
        :return: Nos regresa el estado del canal
                  puede valer 0 (Canal libre)
                  puede valer 1 (Canal ocupado)
                  puede valer 2 (Colisión)
        """
        result = 0

        if self.col:
            if tipo:
                result = -1
        else:
            result = self.n_trans

        return result

    def ocupar(self):
        self.n_trans += 1
        if self.n_trans > 1:
            self.col = True

    def desocupar(self):
        self.n_trans -= 1
        if self.n_trans == 0:
            self.col = False

    def reset(self):
        self.n_trans = 0
        self.col = False
